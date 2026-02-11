# Mat processing
from oja.oja_rl import OjaRL
import gymnasium as gym
import numpy as np
import random

# Threading processing
import time
import multiprocessing
import logging
import uuid
import itertools

# Logging config
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
def run_env(model, env):
    observation, info = env.reset()
    done = False
    rewards = 1
            
    while not done:
        action = model(observation, rewards)
        observation, reward, terminated, truncated, info = env.step(action)
        rewards += reward

        done = terminated or truncated
        if done:
            logging.info(f'Thread {model.model_id} ended: reward={rewards}')

    env.close()
    return model, rewards

class EnvTrainer:
    # Number of individuals = n_features^2
    def __init__(self, model=OjaRL, n_features:int=10, noise=.5, env=gym.make('CartPole-v1')):
        self.min_lr, self.max_lr = 0, 1
        self.min_discount_factor, self.max_discount_factor = 0, 1
        self.n_features = n_features
        self.model = model
        self.noise = noise
        self.chosen_first = None
        self.std_weight = np.random.rand(4, 1)
        self.env = env

    def generate_models(self):
        self.current_lr = np.linspace(self.min_lr, self.max_lr, num=self.n_features)
        self.current_discount_factor = np.linspace(self.min_discount_factor, self.max_discount_factor, num=self.n_features)
        self.models = list(map(lambda x: self.model(*x), itertools.product(self.current_lr, self.current_discount_factor)))
        
        for model in self.models:
            model.W = self.std_weight

    def run_models_and_choice(self):        
        args = [(model, self.env) for model in self.models]
        with multiprocessing.Pool(processes=5) as pool:
            models, rewards = zip(*pool.starmap(run_env, args))

        #time.sleep(1)
        max_up = 1
        max_d = 0

        max_up_idx = 0
        max_d_idx = 0

        for i, r in enumerate(rewards):
            if r > max_up:
                max_up = r
                max_up_idx = i
            
            elif max_up > r > max_d:
                max_d = r
                max_d_idx = i

        
        rewards_chosen_first, rewards_chosen_second = rewards[max_up_idx], rewards[max_d_idx]
        # Get parameters
        self.min_lr, self.max_lr = sorted([models[max_d_idx].learning_rate, models[max_up_idx].learning_rate])
        self.min_lr *= self.noise
        self.max_lr /= self.noise

        self.min_discount_factor, self.max_discount_factor = sorted([models[max_d_idx].discount_factor, models[max_up_idx].discount_factor])
        self.min_discount_factor *= self.noise
        self.max_discount_factor /= self.noise
        
        self.std_weight = models[max_up_idx].W
        self.chosen_first = models[max_up_idx]

        logging.info(f'Best model parameters: W = {models[max_up_idx].W}; lr={models[max_up_idx].learning_rate}; discount_factor={models[max_up_idx].discount_factor}') 
        logging.info(f'Rewards: {rewards_chosen_first} / {rewards_chosen_second}')

    def run(self, episodes:int):
        for ep in range(episodes):
            self.generate_models()
            self.run_models_and_choice()
            logging.info(f'EPOCH {ep+1} finalized!')             
        
        return self.chosen_first
    