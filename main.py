from oja.oja_rl import OjaRL
from environ import test_env_peaks_with_logging
import gymnasium as gym

if __name__ == '__main__':

    env = gym.make('CartPole-v0')
    ac = env.action_space.n
    o = env.observation_space.shape[0]

    print(env.observation_space.shape)
    model = OjaRL(learning_rate=2, discount_factor=0.9, n_actions=ac, n_features=o)

    test_env_peaks_with_logging(model, env)
