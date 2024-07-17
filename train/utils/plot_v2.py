import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
import seaborn as sns
import os
import numpy as np
from matplotlib.ticker import ScalarFormatter
try:
    from tensorflow.python.training.summary_io import summary_iterator
    # we need tensorflow for extracting data from tensorboard events.
except:
    pass
try:
    from eval_v2 import eval
except:
    pass

labels = [ 'info/evaluation',] # 'info/eval_ret', 'info/evaluation'

sns.set(style='whitegrid', font_scale=2.0, rc={'font.family': 'STIXGeneral',
                                               'mathtext.fontset': 'stix'})

def extract_data_from_events(path, tags):
    data = {key : [] for key in tags+['training_iteration']}
    try:
        for e in summary_iterator(path):
            for v in e.summary.value:
                if v.tag in tags :
                    # try:
                    data.get('training_iteration').append(int(e.step / 1e4))
                    data.get(v.tag).append(v.simple_value)
    except:
        pass

    return pd.DataFrame.from_dict(data)

def plot(data_source = 'events'):
    # sns.set(style='darkgrid', font_scale=1.3)
    sns.set_palette([(0.0, 0.24705882352941178, 1.0),
                     (0.011764705882352941, 0.9294117647058824, 0.22745098039215686),
                     (0.9098039215686274, 0.0, 0.043137254901960784),
                     (0.5411764705882353, 0.16862745098039217, 0.8862745098039215),
                     (1.0, 0.7686274509803922, 0.0),
                     (0.0, 0.8431372549019608, 1.0)])

    title = 'Training'
    hue = 'Algorithm'
    path_dict = {
        'random feature': '/home/mht/sim_to_real/training/log/hover-aviary-v0/sac',
    }

    dfs = []
    for key, path in path_dict.items():
        # rfdim = int(rfdim)
        for name_dir in os.listdir(path):
            name_dir_path = os.path.join(path, name_dir)
            for seed_dir in os.listdir(name_dir_path):
                if not os.path.isdir(os.path.join(name_dir_path, seed_dir)):
                    continue
                if seed_dir.startswith('skip'):
                    continue
                abs_path = os.path.join(name_dir_path, seed_dir)
                if data_source == 'csv':
                    df = pd.read_csv(os.path.join(name_dir_path, 'progress.csv'))
                elif data_source == 'events':
                    for fname in os.listdir(abs_path):
                        if fname.startswith('events'):
                            break
                    df = extract_data_from_events(os.path.join(abs_path, fname), labels)


                df[hue] = key
                dfs.append(df)

    total_df = pd.concat(dfs, ignore_index=False)
    for y in labels: #  'episode_reward_max',
        plt.figure(figsize=[6, 4])
        sns.lineplot(total_df, x='training_iteration', y=y, hue=hue, palette='muted', legend=False)
        # plt.tight_layout()
        # title = ' Pendubot'
        plt.title(title)
        plt.ylabel('episodic return')
        # plt.xlim(0, 300000)
        if title == '2D Drones':
            plt.ylim(-200, 0)
        plt.xlabel(r'training samples ($\times 10^4$)')
        # plt.ticklabel_format(axis='x', style='scientific')
        # formatter = ScalarFormatter(useMathText=True)
        # formatter.set_scientific(True)
        # plt.gca().xaxis.set_major_formatter(formatter)
        plt.tight_layout()
        # plt.show()
        figpath = '/home/mht/sim_to_real/training/log/figure/' + title + y.split('/')[1] + '.pdf'
        plt.savefig(figpath)

def plot_bar(from_data = False):
    sns.set(style='whitegrid', font_scale=2.0, rc={'font.family': 'STIXGeneral'})
    # List of colors
    colors = [ 'blue', 'orange','green', 'red']

    # Create a ListedColormap
    custom_cmap = ListedColormap(colors)

    title = '2D Drones Noisy Nystrom'
    path_dict = {
        '2048': '/home/mht/PycharmProjects/lvrep-rl-cloned/log/Quadrotor2D-v2_sigma_1.0_rew_scale_10.0/rfsac_nystrom_True_rf_num_2048_sample_dim_8192',
        '4096': '/home/mht/PycharmProjects/lvrep-rl-cloned/log/Quadrotor2D-v2_sigma_1.0_rew_scale_10.0/rfsac_nystrom_True_rf_num_2048_sample_dim_4096',
        '8192': '/home/mht/PycharmProjects/lvrep-rl-cloned/log/Quadrotor2D-v2_sigma_1.0_rew_scale_10.0/rfsac_nystrom_True_rf_num_4096_sample_dim_4096',
        # '20482': '/home/mht/PycharmProjects/lvrep-rl-cloned/log/Quadrotor2D-v2_sigma_1.0_rew_scale_10.0/rfsac_nystrom_True_rf_num_2048_sample_dim_4096',
        # '40962': '/home/mht/PycharmProjects/lvrep-rl-cloned/log/Quadrotor2D-v2_sigma_1.0_rew_scale_10.0/rfsac_nystrom_True_rf_num_4096_sample_dim_4096',
        # '81922': '/home/mht/PycharmProjects/lvrep-rl-cloned/log/Quadrotor2D-v2_sigma_1.0_rew_scale_10.0/rfsac_nystrom_False_rf_num_4096_learn_rf_True',
        # '81921': '/home/mht/PycharmProjects/lvrep-rl-cloned/log/Quadrotor2D-v2_sigma_1.0_rew_scale_10.0/rfsac_nystrom_False_rf_num_8192_learn_rf_True',
        # '20481': '/home/mht/PycharmProjects/lvrep-rl-cloned/log/Quadrotor2D-v2_sigma_1.0_rew_scale_10.0/rfsac_nystrom_True_rf_num_2048_sample_dim_8192',
        # '40961': '/home/mht/PycharmProjects/lvrep-rl-cloned/log/Quadrotor2D-v2_sigma_1.0_rew_scale_10.0/rfsac_nystrom_True_rf_num_2048_sample_dim_4096',

    }

    # title = '2D Drones Random Feature'
    # path_dict = {
    #     '2048': '/home/mht/PycharmProjects/lvrep-rl-cloned/log/Quadrotor2D-v2_sigma_1.0_rew_scale_10.0/rfsac_nystrom_True_rf_num_2048_sample_dim_8192',
    #     '4096': '/home/mht/PycharmProjects/lvrep-rl-cloned/log/Quadrotor2D-v2_sigma_1.0_rew_scale_10.0/rfsac_nystrom_True_rf_num_2048_sample_dim_4096',
    #     '8192': '/home/mht/PycharmProjects/lvrep-rl-cloned/log/Quadrotor2D-v2_sigma_1.0_rew_scale_10.0/rfsac_nystrom_True_rf_num_4096_sample_dim_4096',
    # }

    # title = 'Pendubot Random Feature'
    # path_dict = {
    #     '2048': '/home/mht/PycharmProjects/lvrep-rl-cloned/log/Pendubot-v0_sigma_0.0_rew_scale_3.0_reward_energy/rfsac_nystrom_False_rf_num_2048_learn_rf_True',
    #     '4096': '/home/mht/PycharmProjects/lvrep-rl-cloned/log/Pendubot-v0_sigma_0.0_rew_scale_3.0_reward_energy/rfsac_nystrom_False_rf_num_4096_learn_rf_True',
    #     '8192': '/home/mht/PycharmProjects/lvrep-rl-cloned/log/Pendubot-v0_sigma_0.0_rew_scale_3.0_reward_energy/rfsac_nystrom_False_rf_num_8192_learn_rf_True'
    # }

    # title = 'Pendubot Random Feature'
    # path_dict = {
    #     '2048': '/home/mht/PycharmProjects/lvrep-rl-cloned/log/Pendubot-v0_sigma_0.0_rew_scale_3.0_reward_energy/rfsac_nystrom_False_rf_num_2048_learn_rf_False',
    #     '4096': '/home/mht/PycharmProjects/lvrep-rl-cloned/log/Pendubot-v0_sigma_0.0_rew_scale_3.0_reward_energy/rfsac_nystrom_False_rf_num_4096_learn_rf_False',
    #     '8192': '/home/mht/PycharmProjects/lvrep-rl-cloned/log/Pendubot-v0_sigma_0.0_rew_scale_3.0_reward_energy/rfsac_nystrom_False_rf_num_8192_learn_rf_False'
    # }

    # title = 'Noisy Pendubot Random Feature'
    # path_dict = {
    #     '2048': '/home/mht/PycharmProjects/lvrep-rl-cloned/log/Pendubot-v0_sigma_1.0_rew_scale_3.0_reward_energy/rfsac_nystrom_False_rf_num_2048_learn_rf_False',
    #     '4096': '/home/mht/PycharmProjects/lvrep-rl-cloned/log/Pendubot-v0_sigma_1.0_rew_scale_3.0_reward_energy/rfsac_nystrom_False_rf_num_4096_learn_rf_False',
    #     '8192': '/home/mht/PycharmProjects/lvrep-rl-cloned/log/Pendubot-v0_sigma_1.0_rew_scale_3.0_reward_energy/rfsac_nystrom_False_rf_num_8192_learn_rf_False'
    # }


    # title = 'Pendubot'
    # path_dict = {
    #     'Top 1024 of 1024' : '/media/mht/新加卷/lvrep_log/Pendubot-v0_sigma_0.0_rew_scale_3.0_reward_lqr/rfsac_nystrom_True_rf_num_1024_sample_dim_1024',
    #     # TODO: outliers
    #     'Top 1024 of 2048' : '/media/mht/新加卷/lvrep_log/Pendubot-v0_sigma_0.0_rew_scale_3.0_reward_lqr/rfsac_nystrom_True_rf_num_1024_sample_dim_2048',
    #     'Top 1024 of 4096' : '/media/mht/新加卷/lvrep_log/Pendubot-v0_sigma_0.0_rew_scale_3.0_reward_lqr/rfsac_nystrom_True_rf_num_1024_sample_dim_4096'
    # }
    #
    # title = 'Noisy Pendubot'
    # path_dict = {
    #     'Top 1024 of 1024': '/media/mht/新加卷/lvrep_log/Pendubot-v0_sigma_1.0_rew_scale_3.0_reward_lqr/rfsac_nystrom_True_rf_num_1024_sample_dim_1024',
    #     'Top 1024 of 2048': '/media/mht/新加卷/lvrep_log/Pendubot-v0_sigma_1.0_rew_scale_3.0_reward_lqr/rfsac_nystrom_True_rf_num_1024_sample_dim_2048',
    #     'Top 1024 of 4096': '/media/mht/新加卷/lvrep_log/Pendubot-v0_sigma_1.0_rew_scale_3.0_reward_lqr/rfsac_nystrom_True_rf_num_1024_sample_dim_4096'
    # }

    if not from_data:
        dfs = []
        for key, path in path_dict.items():
            best_mean = -1e6
            best_ep_rets = None
            for dir in os.listdir(path):
                if not os.path.isdir(os.path.join(path, dir)):
                    continue
                try:
                    ep_rets = eval(os.path.join(path, dir))
                except:
                    continue
                mean = np.mean(ep_rets)
                if mean > best_mean:
                    best_mean = mean
                    best_ep_rets = ep_rets
            df = pd.DataFrame.from_dict({'ep_rets': [best_ep_ret / 2.1 for best_ep_ret in best_ep_rets]}) # / 1.8
            df['Algorithm'] = key
            dfs.append(df)

        total_df = pd.concat(dfs, ignore_index=True)
    else:
        total_df = pd.read_csv('/home/mht/PycharmProjects/lvrep-rl-cloned/utils/data_plot/rf_bar_data.csv')
    plt.figure(figsize=[6, 4])
    if not from_data:
        total_df.to_csv('/home/mht/PycharmProjects/lvrep-rl-cloned/utils/data_plot/noisy_nystrom_bar_data.csv')
    plt.grid(zorder=0)
    ax = sns.barplot(total_df, x='Algorithm', y='ep_rets', palette='muted')
    for patch in ax.patches:
        patch.set_zorder(2)  # You can set any zorder value you need
    plt.title(title) # , fontsize = "x-large"
    plt.ylabel('')

    plt.xlabel('feature dimensions') # , fontsize = "x-large"
    plt.tight_layout()
    # plt.show()
    figpath = '/home/mht/PycharmProjects/lvrep-rl-cloned/fig/' + title + '_bar.pdf'
    plt.savefig(figpath)



def plot_pendulum():
    sns.set(style='darkgrid', font_scale=1)
    sns.set_palette([(0.0, 0.24705882352941178, 1.0),
                     (0.011764705882352941, 0.9294117647058824, 0.22745098039215686),
                     (0.9098039215686274, 0.0, 0.043137254901960784),
                     (0.5411764705882353, 0.16862745098039217, 0.8862745098039215),
                     (1.0, 0.7686274509803922, 0.0),
                     (0.0, 0.8431372549019608, 1.0)])
    path_dict = {
        'Random Feature SAC': '/home/mht/ray_results/SAC_Pendulum-v1_2023-04-24_09-36-31jrzl7hdp',
        'SAC' : '/home/mht/ray_results/SAC_Pendulum-v1_2023-04-23_19-18-33qzefa_7_',
        # '16384': '/home/mht/ray_results/RFSAC_Quadrotor-v1_2023-05-08_09-18-428jl_v2ly',
        # '32768': '/home/mht/ray_results/RFSAC_Quadrotor-v1_2023-05-08_18-58-048lea_yvt'
    }

    dfs = []
    for rfdim, path in path_dict.items():
        # rfdim = int(rfdim)
        df = pd.read_csv(os.path.join(path, 'progress.csv'))
        df['algorithm'] = rfdim
        a = 0
        dfs.append(df)

    total_df = pd.concat(dfs, ignore_index=True)
    for y in ['episode_reward_mean', ]: # 'episode_reward_min', 'episode_reward_max', 'episode_len_mean'
        plt.figure(figsize=[6, 4])
        sns.lineplot(total_df, x='training_iteration', y=y, hue='algorithm', palette='muted')
        plt.tight_layout()
        plt.xlim([-2, 500])
        plt.title('Mean episodic return')
        plt.ylabel('')
        plt.xlabel('training iterations')
        plt.tight_layout()
        # plt.show()
        figpath = '/home/mht/PycharmProjects/rllib_random_feature/fig/pen_' + y + '.png'
        plt.savefig(figpath)

if __name__ == '__main__':
    plot()
