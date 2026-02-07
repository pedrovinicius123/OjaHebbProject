import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import softmax

class OjaRL:
    def __init__(self, **hparams):
        self.hparams = {**hparams}
        self.Q_sa = np.random.rand(self.hparams['n_actions'])
        self.probs = softmax(self.Q_sa)
        self.rewards = []

        self.discount_factor = self.hparams['discount_factor']
        self.learning_rate = self.hparams['learning_rate']
        self.VS = .0

        def calc_VS(reward:float):
            chosen = random.choices(range(self.hparams['n_actions']), weights=self.probs.reshape(-1), k=1)[0]
            forward = np.sum(self.probs * self.Q_sa) + reward
            sum_a = np.sum([prob for i, prob in enumerate(self.probs) if i != chosen])
            forward = forward.reshape(forward.size, 1)

            self.VS *= forward * self.discount_factor * sum_a
            self.VS = np.clip(self.VS, -1e10, 1e10)

            return chosen

        def update_Q_sa(chosen:int, reward:float):
            prob_cp = self.probs.reshape(self.probs.size, 1)
            delta_Q_sa = self.VS * self.learning_rate * (prob_cp - self.VS * self.Q_sa)
            self.Q_sa = (reward * self.discount_factor + delta_Q_sa) * self.probs[chosen]
            self.probs = softmax(np.sum(self.Q_sa, axis=1))


        self.calc_VS = calc_VS
        self.update_Q_sa = update_Q_sa

    def forward_learn(self, reward:float):
        chosen = self.calc_VS(reward)
        self.update_Q_sa(chosen, reward)
        self.rewards.append(reward)
        return chosen

    def show_rewards(self):
        plt.plot(range(len(self.rewards)), self.rewards)
        plt.show()
