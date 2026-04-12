from oja.oja_rl import OjaRL
from utils.manager import EnvTrainer
from environ_cartpole import test, record_videos
import gymnasium as gym
import numpy as np

if __name__ == '__main__':
    n_logs = 20
    trainer = EnvTrainer((-float("inf"), 0), (0, float("inf")), n_features=14)
    
    #record_videos(model)
