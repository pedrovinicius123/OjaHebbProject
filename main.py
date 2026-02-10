from oja.oja_rl import OjaRL
from environ import test
import gymnasium as gym

if __name__ == '__main__':

    env = gym.make('CartPole-v1')
    ac = env.action_space.n
    o = env.observation_space.shape[0]
    env.close()

    print(env.observation_space.shape)
    model = OjaRL(learning_rate=0.005, discount_factor=0.95, n_actions=ac, n_features=o, a_plus=30, a_minus=10)
    test(model)
