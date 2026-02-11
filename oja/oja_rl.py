# Threading
import uuid
import time

# Mat processing
import numpy as np
from scipy.special import softmax, expit

scaler = lambda x: softmax(x/0.3)
def alpha(x:np.ndarray):
    cp = x.copy()
    cp[cp > 0] = 1
    cp[cp <= 0] = -1

    return cp

class OjaRL:
    def __init__(self, model_id=uuid.uuid4(), learning_rate:float=.01, discount_factor=.95, n_features=4):
        self.rewards = 1
        self.model_id = model_id 
        self.W = np.random.rand(n_features, 1)

        self.temp = 0.3
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        
    def __call__(self, obs, reward:float, frozen:bool=False) -> int:
        obs = obs.reshape(1, obs.size)
        output = obs @ self.W
        chosen = 1 if output[0] > 0 else 0

        self.rewards += 1/reward
        #print(self.rewards, self.W, output)
        
        delta_W = self.learning_rate * output[0] * (obs.T - output[0] * self.W * (1 if output[0] > 0 else -1)) * self.rewards if not frozen else .0
        self.W -= delta_W
        self.W = scaler(self.W)
        #print(self.W, output)
        #print(chosen)
        return chosen

