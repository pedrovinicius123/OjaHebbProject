from oja.oja_rl import OjaRL
from environ import test_env
import gymnasium as gym

if __name__ == '__main__':

    env = gym.make('CartPole-v0')
    ac = env.action_space.n


    print(env.observation_space.shape)
    model = OjaRL(learning_rate=0.5, discount_factor=0.9, n_actions=ac)

    test_env(model, env)
