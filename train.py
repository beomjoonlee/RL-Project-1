"""
# 학습 코드

## env.step 결과
observation:
  Robot: array(5)
    distance
    v_pref
    velocity_x
    velocity_y
    radius

  human1~5: array(7)
    distance
    position_vector_x
    position_vector_y
    velocity_vector_x
    velocity_vector_y
    radius
    human_radius + robot_radius


info:
  distance: float

"""
import gym_examples
import gymnasium as gym
import time

env = gym.make('gym_examples/GridWorld-v0')
env.action_space.seed(42)

episode = 20000
episode_num = 1

observation, info = env.reset(seed=episode_num)

while True:

    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
    # print(observation)

    if terminated or truncated:
        episode_num += 1
        if episode_num <= episode:
            observation, info = env.reset(seed=episode_num)
        else:
            break

env.close()
