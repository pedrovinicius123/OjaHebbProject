from oja.oja_rl import OjaRL
from utils.manager import EnvTrainer
from environ_cartpole import test, record_videos
from environ_moonlander import test_moonlander
import gymnasium as gym
import numpy as np

if __name__ == '__main__':
    n_logs = 20
    trainer = EnvTrainer((-float("inf"), 0), (0, 1), (1, 2), (2, 3), (3, float("inf")), env=gym.make("LunarLander-v3"), n_features=14, n_input_features=8)
    for i in range(n_logs):
        model = trainer.run(10, i)
        model.learning_rate = 0
        test(model)
    #record_videos(model)
