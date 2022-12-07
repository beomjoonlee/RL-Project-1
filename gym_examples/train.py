import gym_examples
import gymnasium as gym
import time

env = gym.make('gym_examples/GridWorld-v0')
env.action_space.seed(42)

observation, info = env.reset(seed=42)

for _ in range(1000):
    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
    
    time.sleep(0.25)


    if terminated or truncated:
        observation, info = env.reset()

env.close()