import collections
import numpy as np
import torch
import socket

from train.utils import cfusdlog
import matplotlib.pyplot as plt
import re
import argparse
import seaborn as sns
import pandas as pd
import os
from scipy.spatial.transform import Rotation


Batch = collections.namedtuple(
	'Batch',
	['state', 'action', 'reward', 'next_state', 'done']
	)


class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6), device='cpu'):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.done = np.zeros((max_size, 1))

		device_name = socket.gethostname()
		if device_name.startswith('naliseas'):
			from train import CUDA_DEVICE_WORKSTATION
			self.device = torch.device(CUDA_DEVICE_WORKSTATION if torch.cuda.is_available() else "cpu")
		else:
			self.device = torch.device(device)
		

	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.done[self.ptr] = done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return Batch(
			state=torch.FloatTensor(self.state[ind]).to(self.device),
			action=torch.FloatTensor(self.action[ind]).to(self.device),
			next_state=torch.FloatTensor(self.next_state[ind]).to(self.device),
			reward=torch.FloatTensor(self.reward[ind]).to(self.device),
			done=torch.FloatTensor(self.done[ind]).to(self.device),
		)

class RealDataBuffer(ReplayBuffer):

	def __init__(self, max_size=int(1e6), device='cpu'):
		super(RealDataBuffer, self).__init__(state_dim = 28, action_dim = 4, max_size=max_size,
											 device=device)

	def compute_reward(self, pos_error, rpy, vxyz, rpy_rate, control):
		rew_pos = - 2.5 * np.linalg.norm(pos_error, axis=1)
		rew_rpy = - 0.1 * np.linalg.norm(rpy, axis=1)
		rew_lin_vel = - 0.05 * np.linalg.norm(vxyz, axis=1)
		rew_ang_vel = - 0.05 * np.linalg.norm(rpy_rate, axis=1)
		rew_action = - 0.1 * np.linalg.norm(control, axis=1)
		# rew_action_diff = -0.
		# self.rew_info = {'rew_pos': rew_pos,
		# 				 'rew_rpy': rew_rpy,
		# 				 'rew_lin_vel': rew_lin_vel,
		# 				 'rew_ang_vel': rew_ang_vel,
		# 				 'rew_action': rew_action,
		# 				 'rew_action_diff': rew_action_diff
		# 				 }
		return 2 + (rew_pos +
					rew_rpy +
					rew_lin_vel +
					rew_ang_vel +
					rew_action # +
					# rew_action_diff
					)

	def load_usd_data(self, filename, goal=np.array([0.0, 0.0, 1.2])):
		# decode binary log data
		rawData = cfusdlog.decode(filename)
		rawData = rawData['fixedFrequency']
		index = rawData['motor.m1req'] > 0
		first_index = np.nonzero(index)[0][0]
		# state
		# initial_x, initial_y = rawData['stateEstimateZ.x'][first_index] / 1000, rawData['stateEstimateZ.y'][first_index] / 1000
		# goal[0] = initial_x
		# goal[1] = initial_y
		pos_error = -1 * np.vstack([
						 rawData['ctrlMel.pos_error_x'][index],
						 rawData['ctrlMel.pos_error_y'][index],
						 rawData['ctrlMel.pos_error_z'][index],
						 ]).T#  + goal
		integral_pos_error = np.zeros_like(pos_error)
		diff_pos_error = np.zeros_like(pos_error)
		for i in range(pos_error.shape[0]):
			integral_pos_error[i] = np.sum(pos_error[:i+1], axis=0)
		for i in range(1, pos_error.shape[0]):
			diff_pos_error[i] = (pos_error[i] - pos_error[i - 1]) * 240
		rpy = np.vstack([rawData['stabilizer.roll'][index],
						 rawData['stabilizer.pitch'][index],		# all the logging is crazyflie yaw coordination
						 rawData['stabilizer.yaw'][index]]).T / 180. * np.pi
		quat = np.zeros([rpy.shape[0], 4])
		for i, angle in enumerate(rpy):
			quat[i] = Rotation.from_euler('XYZ', np.multiply([1, -1, 1], angle)).as_quat()
		vxyz = np.vstack([rawData['stateEstimate.vx'][index],
						  rawData['stateEstimate.vy'][index],
						  rawData['stateEstimate.vz'][index]]).T
		rpy_rate = np.vstack([rawData['stateEstimateZ.rateRoll'][index],
							  rawData['stateEstimateZ.ratePitch'][index],
							  rawData['stateEstimateZ.rateYaw'][index]]).T / 1000.  # rate in milliradians
		integral_rpy_error = np.vstack([rawData['ctrlMel.i_err_mx'][index],
										rawData['ctrlMel.i_err_my'][index],
							  			rawData['ctrlMel.i_err_mz'][index]]).T  # rate in milliradians
		ctrl = np.vstack([rawData['motor.m1req'][index] / 65535.,
						  rawData['motor.m2req'][index] / 65535.,
						  rawData['motor.m3req'][index] / 65535.,
						  rawData['motor.m4req'][index] / 65535.]).T
		state = np.hstack([pos_error, quat, rpy, vxyz, rpy_rate,
						   integral_pos_error,			# integral
						   -1 * vxyz,					# diff
						   integral_rpy_error,			# integral
						   -1 * rpy_rate,				# integral
						   ])
		st = state[:-1]
		at = ctrl[:-1]
		stp1 = state[1:]
		reward = self.compute_reward(pos_error, rpy, vxyz, rpy_rate, ctrl)[:-1]
		return st, at, reward, stp1

	def load_all_data(self, log_path):
		sts, ats, rewards, stp1s = [], [], [], []
		for log_file in os.listdir(log_path):
			st, at, reward, stp1 = self.load_usd_data(os.path.join(log_path, log_file))
			sts.append(st)
			ats.append(at)
			rewards.append(reward)
			stp1s.append(stp1)
		self.state = np.vstack(sts)
		self.action = np.vstack(ats)
		self.reward = np.hstack(rewards)[:, np.newaxis]
		self.next_state = np.vstack(stp1s)
		self.done = np.zeros_like(self.reward)
		self.size = self.state.shape[0]


def test_load_single_data():
	buf = RealDataBuffer()
	st, at, reward, stp1 = buf.load_usd_data('/media/naliseas-workstation/crazyflie/log27')
	print(f"{st.shape} {at.shape} {reward.shape} {stp1.shape}")

def test_replay():
	buf = RealDataBuffer()
	buf.load_all_data('/home/naliseas-workstation/Documents/haitong/sim_to_real/quad_sim/deploy/sample_log')
	print(buf.sample(batch_size=256))


