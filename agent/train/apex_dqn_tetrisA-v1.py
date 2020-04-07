import logging
import numpy as np

import ray
from ray.rllib.agents.dqn.apex import ApexTrainer
from ray.rllib.agents.dqn.apex import APEX_DEFAULT_CONFIG as DEFAULT_CONFIG

from ..config import NUM_CPU, NUM_GPU
from ..env import *


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Start up Ray. This must be done before we instantiate any RL agents.
ray.init(num_gpus=NUM_GPU, webui_host='127.0.0.1')

config = DEFAULT_CONFIG.copy()

config['log_level'] = 'INFO'
config['ignore_worker_failures'] = True
config['num_workers'] = NUM_CPU
config['num_gpus'] = NUM_GPU

# Raw output data [240, 256, 3]
config['model']['dim'] = 84

# out_size, kernel, stride
filters_84x84 = [
    [16, [8, 8], 4],
    [32, [4, 4], 2],
    [256, [11, 11], 1],
]
config['model']["conv_filters"] = filters_84x84


config['min_iter_time_s'] = 5
config['n_step'] = 2
config['target_network_update_freq'] = 0
config['timesteps_per_iteration'] = 50000
config['train_batch_size'] = 128
config['lr'] = 0.0050

# === Evaluation ===
config['evaluation_interval'] = 50
config['evaluation_num_episodes'] = 5


agent = ApexTrainer(config, "TetrisA-v2")

reward = -999
epoch = 0

while reward < 200:
    result = agent.train()
    print(f'=========== RESULT {epoch} =================')
    result = dict(result)
    print(result)

    reward = result['episode_reward_mean']
    if np.isnan(reward):
        reward = -999

    # Move to next epoch
    epoch += 1
