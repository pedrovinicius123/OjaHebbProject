import random
import numpy as np
from scipy.special import softmax, expit
import time

class OjaRL:
    def __init__(self, **hparams):
        self.hparams = {**hparams}
        self.probs = softmax(np.array([[.5], [.5]]))
        self.rewards = .0
        
        self.a_plus = self.hparams['a_plus']
        self.a_minus = self.hparams['a_minus']
        self.W = np.random.rand(self.hparams['n_features'], 1)

        self.temp = 0.3
        self.discount_factor = self.hparams['discount_factor']
        self.learning_rate = self.hparams['learning_rate']
        self.scaler = lambda x: softmax(x/self.temp)

        def alpha(x:np.ndarray):
            cp = x.copy()
            cp[cp > 0] = 1
            cp[cp <= 0] = -1

            return cp

        self.alpha = alpha

    def forward_learn(self, obs, reward:float) -> int:
        obs = obs.reshape(1, obs.size)
        output = obs @ self.W
        chosen = 1 if output[0] > 0 else 0

        self.rewards += 1/reward
        
        delta_W = self.learning_rate * output[0] * (obs.T - output[0] * self.W * (1 if output[0] > 0 else -1)) * self.rewards
        self.W -= delta_W
        self.W = self.scaler(self.W)
        print(self.W, output, self.probs)
        print(chosen)
        return chosen

