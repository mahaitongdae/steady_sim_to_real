import numpy as np
import torch
import math
from torch import nn
import torch.nn.functional as F
from torch import distributions as pyd
import sys
sys.path.append('/home/naliseas-workstation/Documents/haitong/sim_to_real/quad_sim')
from train.utils import util
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType
import socket
from train.agent.sac.actor import SquashedNormal


device_name = socket.gethostname()
if device_name.startswith('naliseas'):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DifferentiableMellinger(nn.Module):
    cf_mass = 0.027
    massThrust = 132000
    INT16_MAX = 65536
    PWM2RPM_SCALE = 0.2685
    PWM2RPM_CONST = 4070.3
    MIN_PWM = 20000
    MAX_PWM = 65535
    GRAVITY = 9.81 * cf_mass
    KF = 3.16e-10

    def __init__(self, max_rpm=21600, ctrl_freq : int = 240, output = "pwm"):
        """

        Args:
            max_rpm: maximum of rpm it can achieve, useless of we output pwm
            ctrl_freq: Control frequency in Hz
            output:     "pwm" or "rpm",
        """
        super().__init__()
        self.CTRL_FREQ = ctrl_freq
        self.MAX_RPM = max_rpm
        self.integral_error = torch.zeros([3, ])

        self.kp_xy = torch.nn.Parameter(torch.tensor(0.4, requires_grad=True))
        self.ki_xy = torch.nn.Parameter(torch.tensor(0.05, requires_grad=False))
        self.kd_xy = torch.nn.Parameter(torch.tensor(0.2, requires_grad=False))
        # Z position
        self.kp_z = torch.nn.Parameter(torch.tensor(1.25, requires_grad=False))
        self.ki_z = torch.nn.Parameter(torch.tensor(0.05, requires_grad=False))
        self.kd_z = torch.nn.Parameter(torch.tensor(0.4, requires_grad=False))
        # Attitude
        self.kR_xy = torch.nn.Parameter(torch.tensor(70000., requires_grad=True))
        self.kw_xy = torch.nn.Parameter(torch.tensor(0., requires_grad=False))
        self.ki_m_xy = torch.nn.Parameter(torch.tensor(20000., requires_grad=False))

        self.kR_z = torch.nn.Parameter(torch.tensor(60000., requires_grad=False))
        self.kw_z = torch.nn.Parameter(torch.tensor(500., requires_grad=False))
        self.ki_m_z = torch.nn.Parameter(torch.tensor(12000., requires_grad=False))

        # # XY positions
        # self.kp_xy = 0.4
        # self.ki_xy = 0.05
        # self.kd_xy = 0.2
        # # Z position
        # self.kp_z = 1.25
        # self.ki_z = 0.05
        # self.kd_z = 0.4
        # # Attitude
        # self.kR_xy = 70000.
        # self.kw_xy = 0.
        # self.ki_m_xy = 20000.
        # # Z Altitude
        # self.kR_z = 60000.
        # self.kw_z = 500.
        # self.ki_m_z = 12000.
        #
        # self.P_COEFF_FOR = torch.nn.Parameter(torch.tensor([self.kp_xy, self.kp_xy, self.kp_z], requires_grad=True))
        # self.I_COEFF_FOR = torch.nn.Parameter(torch.tensor([self.ki_xy, self.ki_xy, self.ki_z], requires_grad=False))
        # self.D_COEFF_FOR = torch.nn.Parameter(torch.tensor([self.kd_xy, self.kd_xy, self.kd_z], requires_grad=False))
        # self.P_COEFF_TOR = torch.nn.Parameter(torch.tensor([self.kR_xy, self.kR_xy, self.kR_z], requires_grad=True))
        # self.I_COEFF_TOR = torch.nn.Parameter(torch.tensor([self.kw_xy, self.kw_xy, self.kw_z], requires_grad=False))
        # self.D_COEFF_TOR = torch.nn.Parameter(torch.tensor([self.ki_m_xy, self.ki_m_xy, self.ki_m_z], requires_grad=False))

        self.i_range_xy = 2.0
        self.i_range_z = 0.4
        self.i_range_m_xy = 1.0
        self.i_range_m_z = 1500.

        self.target_rpy_rates = torch.zeros([3,]).float().to(torch.device('cuda'))
        self.MIXER_MATRIX = torch.tensor([
            [-.5, -.5, -1],
            [-.5, .5, 1],
            [.5, .5, -1],
            [.5, -.5, 1]
        ]).float()
        self.goal = torch.tensor([0., 0., 1.]).to(device)
        self.target_x_c = torch.tensor([1., 0., 0.]).to(device) # assume target rpy always 0
        self.gravity = torch.tensor([0, 0, self.GRAVITY]).to(device)
        self.output_type = output
        self.reset()

        self.trunk = util.mlp(28, 256, 4,
                              2, hidden_activation=nn.ELU(inplace=True))

        self.log_std_bounds=[-20., 1.]

        def transpose(x):
            return x.T
        self.vec_transpose = torch.vmap(transpose)

    def projection_on_gains(self):
        with torch.no_grad():
            self.kp_xy.clamp_(min=0.1)
            self.kR_xy.clamp_(min=10000.)

    def get_controller_parameters_dict(self):

        return self.state_dict()


    def set_device(self, device):
        self.MIXER_MATRIX = self.MIXER_MATRIX.to(device)
        self.goal = self.goal.to(device)
        self.target_x_c = self.target_x_c.to(device)
        self.gravity = self.gravity.to(device)


    def quaternion_to_matrix(self, quaternions: torch.Tensor) -> torch.Tensor:
        """
        Convert rotations given as quaternions to rotation matrices.

        Args:
            quaternions: quaternions with real part last,
                as tensor of shape (..., 4).

        Returns:
            Rotation matrices as tensor of shape (..., 3, 3).
        """
        i, j, k, r = torch.unbind(quaternions, -1)
        # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
        two_s = 2.0 / (quaternions * quaternions).sum(-1)

        o = torch.stack(
            (
                1 - two_s * (j * j + k * k),
                two_s * (i * j - k * r),
                two_s * (i * k + j * r),
                two_s * (i * j + k * r),
                1 - two_s * (i * i + k * k),
                two_s * (j * k - i * r),
                two_s * (i * k - j * r),
                two_s * (j * k + i * r),
                1 - two_s * (i * i + j * j),
            ),
            -1,
        )
        return o.reshape(quaternions.shape[:-1] + (3, 3))

    def _index_from_letter(self, letter: str) -> int:
        if letter == "X":
            return 0
        if letter == "Y":
            return 1
        if letter == "Z":
            return 2
        raise ValueError("letter must be either X, Y or Z.")

    def _angle_from_tan(self,
            axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool
    ) -> torch.Tensor:
        """
        Extract the first or third Euler angle from the two members of
        the matrix which are positive constant times its sine and cosine.

        Args:
            axis: Axis label "X" or "Y or "Z" for the angle we are finding.
            other_axis: Axis label "X" or "Y or "Z" for the middle axis in the
                convention.
            data: Rotation matrices as tensor of shape (..., 3, 3).
            horizontal: Whether we are looking for the angle for the third axis,
                which means the relevant entries are in the same row of the
                rotation matrix. If not, they are in the same column.
            tait_bryan: Whether the first and third axes in the convention differ.

        Returns:
            Euler Angles in radians for each matrix in data as a tensor
            of shape (...).
        """

        i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
        if horizontal:
            i2, i1 = i1, i2
        even = (axis + other_axis) in ["XY", "YZ", "ZX"]
        if horizontal == even:
            return torch.atan2(data[..., i1], data[..., i2])
        if tait_bryan:
            return torch.atan2(-data[..., i2], data[..., i1])
        return torch.atan2(data[..., i2], -data[..., i1])

    def matrix_to_euler_angles(self, matrix: torch.Tensor, convention = "XYZ") -> torch.Tensor:
        """
        Convert rotations given as rotation matrices to Euler angles in radians.

        Args:
            matrix: Rotation matrices as tensor of shape (..., 3, 3).
            convention: Convention string of three uppercase letters.

        Returns:
            Euler angles in radians as tensor of shape (..., 3).
        """
        if len(convention) != 3:
            raise ValueError("Convention must have 3 letters.")
        if convention[1] in (convention[0], convention[2]):
            raise ValueError(f"Invalid convention {convention}.")
        for letter in convention:
            if letter not in ("X", "Y", "Z"):
                raise ValueError(f"Invalid letter {letter} in convention string.")
        if matrix.size(-1) != 3 or matrix.size(-2) != 3:
            raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
        i0 = self._index_from_letter(convention[0])
        i2 = self._index_from_letter(convention[2])
        tait_bryan = i0 != i2
        if tait_bryan:
            central_angle = torch.asin(
                matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
            )
        else:
            central_angle = torch.acos(matrix[..., i0, i0])

        o = (
            self._angle_from_tan(
                convention[0], convention[1], matrix[..., i2], False, tait_bryan
            ),
            central_angle,
            self._angle_from_tan(
                convention[2], convention[1], matrix[..., i0, :], True, tait_bryan
            ),
        )
        return torch.stack(o, -1)


    def reset(self):
        """Resets the control classes.

        The previous step's and integral errors for both position and attitude are set to zero.

        """
        #### Store the last roll, pitch, and yaw ###################
        self.last_rpy = torch.zeros(3).to(device)
        #### Initialized PID control variables #####################
        self.last_pos_e = torch.zeros(1, 3).to(device)
        self.integral_pos_e = torch.zeros(1, 3).to(device)
        self.last_rpy_e = torch.zeros(3).to(device)
        self.integral_rpy_e = torch.zeros(3).to(device)

    def mellinger_control(self, obs):
        #### OBS SPACE OF SIZE 28
        # first 16: xyz_error 3, quat 4, rpy 3, vel_xyz 3, angle_vel_xyz 3 each
        #### then 12: integral error of pos 3, diff error of pos 3, integral error of angle 3, diff error of angle 3

        P_COEFF_FOR = torch.stack([self.kp_xy, self.kp_xy, self.kp_z])
        I_COEFF_FOR = torch.stack([self.ki_xy, self.ki_xy, self.ki_z])
        D_COEFF_FOR = torch.stack([self.kd_xy, self.kd_xy, self.kd_z])
        P_COEFF_TOR = torch.stack([self.kR_xy, self.kR_xy, self.kR_z])
        I_COEFF_TOR = torch.stack([self.kw_xy, self.kw_xy, self.kw_z])
        D_COEFF_TOR = torch.stack([self.ki_m_xy, self.ki_m_xy, self.ki_m_z])

        # position control
        if len(obs.shape) == 1:
            obs = obs.unsqueeze(0)

        pos_e = self.goal - obs[:, 0:3]
        cur_quat = obs[:, 3:7]
        cur_rpy = obs[:, 7:10]
        cur_vel = obs[:, 10:13]
        integral_pos_error = obs[:, 16:19]
        diff_pos_error = obs[:, 19:22]
        integral_rpy_error = obs[:, 22:25]
        diff_rpy_error = -1 * obs[:, 25:28]
        cur_rotation = self.quaternion_to_matrix(cur_quat)
        vel_e = - cur_vel
        # self.integral_pos_e = self.integral_pos_e + pos_e * 1 / 240
        # self.integral_pos_e = torch.clip(self.integral_pos_e, -2., 2.)
        # self.integral_pos_e[:, 2] = torch.clip(self.integral_pos_e[:, 2], -0.15, .15)
        #### PID target thrust #####################################
        target_thrust = torch.multiply(P_COEFF_FOR, pos_e) \
                        + torch.multiply(I_COEFF_FOR, integral_pos_error) \
                        + torch.multiply(D_COEFF_FOR, vel_e) + self.gravity  # , device=self.de
        scalar_thrust = torch.clamp(torch.vmap(torch.inner)(target_thrust, cur_rotation[:, :, 2]), 0, torch.inf)
        # thrust_pwm = (torch.sqrt(scalar_thrust / (4 * self.KF)) - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE
        thrust_pwm = self.massThrust * scalar_thrust
        target_z_ax = F.normalize(target_thrust, dim=1)
        # target_x_c = torch.tensor([1., 0., 0.]) # assume target rpy always 0
        target_y_ax = F.normalize(torch.vmap(torch.cross, in_dims=(0, None))(target_z_ax, self.target_x_c),
                                  dim=1)  # / torch.norm(torch.cross(target_z_ax, target_x_c))
        target_x_ax = torch.vmap(torch.cross)(target_y_ax, target_z_ax)
        target_rotation_transposed = torch.stack([target_x_ax, target_y_ax, target_z_ax], dim=1)
        target_rotation = torch.permute(target_rotation_transposed, [0, 2, 1])
        #### Target rotation #######################################
        target_euler = self.matrix_to_euler_angles(target_rotation)

        # Altitude control
        # cur_rotation = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        # cur_rpy = np.array(p.getEulerFromQuaternion(cur_quat))
        # target_quat = (Rotation.from_euler('XYZ', target_euler, degrees=False)).as_quat()
        # w, x, y, z = target_quat
        # target_rotation = (Rotation.from_quat([w, x, y, z])).as_matrix()

        rot_matrix_e = (torch.matmul(self.vec_transpose(target_rotation), cur_rotation)
                        - torch.matmul(self.vec_transpose(cur_rotation), target_rotation))
        rot_e = torch.stack([rot_matrix_e[:, 2, 1], rot_matrix_e[:, 0, 2], rot_matrix_e[:, 1, 0]], dim=1)
        self.rot_e = rot_e
        rpy_rates_e = self.target_rpy_rates - (cur_rpy - self.last_rpy) * 240
        self.last_rpy = cur_rpy
        self.integral_rpy_e = self.integral_rpy_e - rot_e / 240
        self.integral_rpy_e = torch.clip(self.integral_rpy_e, -1500., 1500.)
        self.integral_rpy_e[0:2] = torch.clip(self.integral_rpy_e[0:2], -1., 1.)
        #### PID target torques ####################################
        target_torques = - torch.multiply(P_COEFF_TOR, rot_e) \
                         + torch.multiply(D_COEFF_TOR, diff_rpy_error) \
                         + torch.multiply(I_COEFF_TOR, integral_rpy_error)
        target_torques = torch.clip(target_torques, -32000, 32000)
        pwm = thrust_pwm.unsqueeze(1) + torch.matmul(self.MIXER_MATRIX, target_torques.unsqueeze(2)).squeeze()
        pwm = torch.clip(pwm, self.MIN_PWM, self.MAX_PWM)  # .squeeze(dim=-1)
        if self.output_type == "pwm":
            return pwm / self.MAX_PWM
        elif self.output_type == "rpm":
            return (self.PWM2RPM_SCALE * pwm + self.PWM2RPM_CONST) / self.MAX_RPM
        else:
            raise ValueError(f"Invalid output type {self.output_type}.")


    def forward(self, obs):
        control = self.mellinger_control(obs)
        log_std = self.trunk(obs)
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std +
                                                                     1)

        std = log_std.exp()
        dist = SquashedNormal(control, std)
        return dist




def test_quaternion_to_matrix():
    from scipy.spatial.transform import Rotation
    policy = DifferentiableMellinger()
    euler = [10.0, 20.0, 30.0]
    q = Rotation.from_euler('XYZ', euler, degrees=True).as_quat(False, scalar_first=False)
    mat_torch = policy.quaternion_to_matrix(torch.tensor(q))
    mat_scipy = Rotation.from_euler('XYZ', euler, degrees=True).as_matrix()
    print(mat_torch)
    print(mat_scipy)

def test_matrix_to_euler_angles():
    from scipy.spatial.transform import Rotation
    policy = DifferentiableMellinger()
    euler = [10.0, 20.0, 30.0]
    m = Rotation.from_euler('XYZ', euler, degrees=True).as_matrix()
    euler = policy.matrix_to_euler_angles(torch.tensor(m))
    print(euler / np.pi)

def test_mellinger_controller():
    def reformat_to_pid_state(state):
        return np.hstack([state[:3] + np.array([0, 0, 1]), state[3:7], state[10:16]])

    from gym_pybullet_drones.envs.single_agent_rl.HoverAviary import HoverAviary
    from scipy.spatial.transform import Rotation
    import matplotlib.pyplot as plt
    import time
    xyz = []
    import gymnasium as gym
    env = gym.make('hover-aviary-v0', gui=True, act=ActionType.RPM,
                   )
    obs, info = env.reset()
    print(env.MAX_RPM)
    done = False
    # policy = DifferentiableMellinger(max_rpm=env.MAX_RPM)
    pid = DSLPIDControl(drone_model=DroneModel.CF2X)
    pid.MIXER_MATRIX = np.array([
        [-.5, .5, -1],
        [-.5, -.5, 1],
        [.5, -.5, -1],
        [.5, .5, 1]
    ])

    while not done:
        # action = policy(torch.tensor(obs)).detach().numpy()[0]
        action = pid.computeControl(control_timestep= 1 / 240,
                                    cur_pos= obs[:3],
                                    cur_quat= obs[3:7],
                                    cur_vel= obs[10:13],
                                    cur_ang_vel=obs[13:16],
                                    target_pos=np.array([0, 0, 1]),)[0]
        action =  action / env.MAX_RPM
        obs, rew, terminated, truncated, info = env.step(action)

        xyz.append(np.hstack([obs[7:10]]))  # , policy.integral_rpy_e.detach().numpy()
        done = terminated or truncated
        time.sleep(0.01)

    print(info)
    xyz = np.array(xyz)
    plt.plot(xyz)
    plt.legend(['r', 'p', 'y']) # , 'r2', 'p2', 'y2'
    print(xyz.shape)
    plt.show()

def test_batch_mellinger():
   
    from gym_pybullet_drones.envs.single_agent_rl.HoverAviary import HoverAviary
    import matplotlib.pyplot as plt
    import time
    import gymnasium as gym
    env = gym.make('hover-aviary-v0', gui=True, act=ActionType.PWM,
                   initial_xyzs=np.array([[0.01, 0.01, 1.01]]),
                   initial_rpys=np.array([[0.0, 0.0, 0.0]]), )

    obs, _ = env.reset()
    obs1, _ , _, _, _ = env.step(env.action_space.sample())
    batched_obs = torch.tensor([obs, obs1])
    policy = DifferentiableMellinger()
    print(policy(batched_obs))
    print(policy.get_controller_parameters_dict())

def test_pitch_angle():
    def reformat_to_pid_state(state):
        return np.hstack([state[:3] + np.array([0, 0, 1]), state[3:7], state[10:16]])

    from gym_pybullet_drones.envs.single_agent_rl.HoverAviary import HoverAviary
    import matplotlib.pyplot as plt
    import time
    xyz = []
    import gymnasium as gym
    env = gym.make('hover-aviary-v0', gui=False, act=ActionType.PWM, initial_xyzs = np.array([[0.0, 0, 1.0]]),
                   initial_rpys = np.array([[0.0, 0.0, 0.9]])
                   )
    obs, info = env.reset()
    print(env.MAX_RPM)
    done = False
    policy = DifferentiableMellinger(max_rpm=env.MAX_RPM)
    pid = DSLPIDControl(drone_model=DroneModel.CF2X)

    while not done:
        action = policy(torch.tensor(obs)).detach().numpy()[0]
        # action = pid.computeControl(control_timestep= 1 / 240,
        #                             cur_pos= obs[:3],
        #                             cur_quat= obs[3:7],
        #                             cur_vel= obs[10:13],
        #                             cur_ang_vel=obs[13:16],
        #                             target_pos=np.array([0, 0, 1]),)[0]
        # action =  action / env.MAX_RPM
        obs, rew, terminated, truncated, info = env.step(action)

        xyz.append(action) # , policy.integral_rpy_e.detach().numpy()
        done = terminated or truncated
        time.sleep(0.01)

    print(info)
    xyz = np.array(xyz)
    plt.plot(xyz[:100])
    plt.legend(['1', '2', '3', '4'])
    print(xyz.shape)
    plt.show()


def test_yaw_mix():
    def reformat_to_pid_state(state):
        return np.hstack([state[:3] + np.array([0, 0, 1]), state[3:7], state[10:16]])

    from gym_pybullet_drones.envs.single_agent_rl.HoverAviary import HoverAviary
    from scipy.spatial.transform import Rotation
    import matplotlib.pyplot as plt
    import time
    xyz = []
    import gymnasium as gym
    env = gym.make('hover-aviary-v0', gui=True, act=ActionType.RPM,  initial_xyzs = np.array([[0.0, 0, 1.0]]),
                   initial_rpys = np.array([[0.0, 0.0, 0.9]])
                   )
    obs, info = env.reset()
    print(env.MAX_RPM)
    done = False
    # policy = DifferentiableMellinger(max_rpm=env.MAX_RPM)
    pid = DSLPIDControl(drone_model=DroneModel.CF2X)
    pid.MIXER_MATRIX = np.array([
        [-.5, .5, -1],
        [-.5, -.5, 1],
        [.5, -.5, -1],
        [.5, .5, 1]
    ])

    while not done:
        # action = policy(torch.tensor(obs)).detach().numpy()[0]
        action = pid.computeControl(control_timestep=1 / 240,
                                    cur_pos=obs[:3],
                                    cur_quat=obs[3:7],
                                    cur_vel=obs[10:13],
                                    cur_ang_vel=obs[13:16],
                                    target_pos=np.array([0, 0, 1]), )[0]
        action = action / env.MAX_RPM
        obs, rew, terminated, truncated, info = env.step(action)

        xyz.append(np.hstack([obs[7:10]]))  # , policy.integral_rpy_e.detach().numpy()
        done = terminated or truncated
        time.sleep(0.01)

    print(info)
    xyz = np.array(xyz)
    plt.plot(xyz)
    plt.legend(['r', 'p', 'y'])  # , 'r2', 'p2', 'y2'
    print(xyz.shape)
    plt.show()

if __name__ == '__main__':
    test_yaw_mix()
# test_yaw_mix()