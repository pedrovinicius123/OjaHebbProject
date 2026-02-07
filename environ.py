import matplotlib.pyplot as plt
import numpy as np

# CODE TAKEN AND ADAPTED FROM GYMNASIUM DOCUMENTATION: https://gymnasium.farama.org/index.html

def test_env(model, env):
    # Reset the environment to generate the first observation
    _,  info = env.reset(seed=42)
    reward = .0
    rewards = []
    tot = 0
    i = 0

    for _ in range(10000):
        # this is where you would insert your policy
        action = model.forward_learn(reward)

        # step (transition) through the environment with the action
        # receiving the next observation, reward and if the episode has terminated or truncated
        _, reward, terminated, truncated, _ = env.step(action)

        # If the episode has ended then we can reset to start a new episode
        if terminated or truncated:
            env.reset()
            rewards.append(tot)
            tot = 0
            continue

        tot += 1


    plt.plot(range(len(rewards)), rewards)
    print(np.mean(rewards))
    plt.show()
    env.close()