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

action: array(2)
  v_pref
  angle

info:
  distance: float

"""
import collections
import random

import gym_examples
import gymnasium as gym
import time
import torch


buffer_limit = 50000  # size of replay buffer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)  # double-ended queue

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float).to(device),\
               torch.tensor(a_lst).to(device), \
               torch.tensor(r_lst).to(device),\
               torch.tensor(s_prime_lst, dtype=torch.float).to(device), \
               torch.tensor(done_mask_lst).to(device)

    def size(self):
        return len(self.buffer)


def main():
    env = gym.make('gym_examples/GridWorld-v0', render_mode='rgb_array')
    env.action_space.seed(42)

    episode = 20000
    episode_num = 1
    total_reward = 0

    memory = ReplayBuffer()

    state, info = env.reset(seed=episode_num)
    done = False

    while True:
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        # print(observation)

        if terminated or truncated:
            done = True

        done_mask = 0.0 if done else 1.0

        if done:
            episode_num += 1
            if episode_num <= episode:
                state, info = env.reset(seed=episode_num)
            else:
                break

        memory.put((state, action, reward, next_state, done_mask))

        if episode_num % 10 == 0:
            print(f'n_episode: {episode_num}, avg_reward: {total_reward / episode_num:.4f}')

    env.close()


if __name__ == '__main__':
    main()
