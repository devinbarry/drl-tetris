import logging
import numpy as np

import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo import DEFAULT_CONFIG

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


config['rollout_fragment_length'] = 250  # Size of batches collected from each worker.
train_batch_size = config['rollout_fragment_length'] * config['num_workers']
config['train_batch_size'] = train_batch_size  # Number of timesteps collected for each SGD round
config['sgd_minibatch_size'] = 512  # Default is 128.
config['num_sgd_iter'] = 20


agent = PPOTrainer(config, "TetrisA-v2")

reward = -999
epoch = 0

# After 8.9M timesteps (8910 seconds, 2.5 hours), remains stuck in a local minimum with a score of around -16
# Best score of around -13 or -12 is possible early on in training

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
