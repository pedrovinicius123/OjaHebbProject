import random
import numpy as np
from scipy.special import softmax

class OjaRL:
    def __init__(self, **hparams):
        self.hparams = {**hparams}
        self.Q_sa = np.random.rand(self.hparams['n_features'], self.hparams['n_actions'])
        self.probs = softmax(self.Q_sa)
        self.rewards = []

        self.discount_factor = self.hparams['discount_factor']
        self.learning_rate = self.hparams['learning_rate']

        def LT(mtrx:np.ndarray):
            cp = mtrx.copy()
            for i in range(cp.shape[0]):
                for j in range(i):
                    cp[i, j] = 0

            return cp

        def update_Q_sa(obs:np.ndarray,  reward:float):
            #print(self.probs.shape, self.Q_sa.T.shape)
            delta_Q_sa = reward*self.learning_rate*(obs @ self.probs.T - self.Q_sa @ LT(self.probs @ self.probs.T))
            self.Q_sa += delta_Q_sa

        self.update_Q_sa = update_Q_sa

    def forward_learn(self, obs:np.ndarray, reward:float):
        self.probs = obs.reshape(1, obs.size) @ self.Q_sa
        obs = obs.reshape(obs.size, 1)

        #print(np.max(self.probs) - np.min(self.probs))
        self.probs = (self.probs - np.min(self.probs)) / (np.max(self.probs) - np.min(self.probs))
        self.probs = self.probs.T
        print(self.probs.T)
        chosen = random.choices(range(self.hparams['n_actions']), weights=self.probs.reshape(-1), k=1)[0]
        self.update_Q_sa(obs, reward)
        return chosen
