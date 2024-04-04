import gym
from gym.wrappers import RecordVideo
import numpy as np
import time
import os
import io
import base64
from IPython.display import HTML
import glob
import qHat
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import time
from matplotlib.animation import FuncAnimation
from pathlib import Path


# env = gym.make('MountainCar-v0', render_mode='rgb_array')

def epsilon_greedy_policy(q_hat, epsilon, num_actions):
  def policy_estimator(obs):
    action_probs = np.ones(num_actions, dtype=float) * epsilon / num_actions
    q_vals = q_hat.predict(obs)
    max_a = np.argmax(q_vals)
    action_probs[max_a] += (1.0 - epsilon)
    return action_probs
  return policy_estimator

def semi_gradient_n_sarsa(n, eps, q_hat, epsilon, gamma, env):
  scores = []
  for ep in range(eps):
    states = []
    actions = []
    rewards = []

    s = env.reset()[0]
    pi = epsilon_greedy_policy(q_hat, epsilon, env.action_space.n)
    action_probs = pi(s)
    # if np.random.rand() < epsilon:
    #   a = np.random.choice(len(action_probs))
    # else:
    #   a = np.argmax(action_probs)
    a = np.random.choice(np.arange(len(action_probs)), p=action_probs)
    T = float('inf')
    t = 0
    states.append(s)
    actions.append(a)
    rewards.append(0.0)
    while True:
      if t < T:
        observation, reward, terminated, truncated, info = env.step(a)
        # print(observation)
        states.append(observation)
        rewards.append(reward)
        if terminated:
          T = t+1
        else:
          next_action_probs = pi(observation)
          # if np.random.rand() < epsilon:
          #   next_a = np.random.choice(len(next_action_probs))
          # else:
          #   next_a = np.argmax(next_action_probs)
          next_a = np.random.choice(np.arange(len(next_action_probs)), p=next_action_probs)
          actions.append(next_a)
      tow = t - n + 1
      if tow >= 0:
        G = 0
        for i in range(tow+1, min(tow+n, T)+1):
          G += pow(gamma, i - tow - 1) * rewards[i]
        if tow + n < T:
          predicted_val = q_hat.predict(states[tow+n], actions[tow+n])
          # print(pow(gamma, n) * predicted_val[0])
          G += gamma**n * predicted_val[0]
          # print(G)
        q_hat.train(states[tow], actions[tow], G)
      if tow == T-1:
        break
      s = observation
      a = next_a
      t += 1
    scores.append(t)
    # print(t)
  env.close()
  return scores

def train_sarsa(env, name_prefix, eps, n, epsilon, gamma, step_size, tilings):
    env._max_episode_steps = 10000
    q_hat = qHat.QHat_Function(tilings, step_size, env)
    scores = semi_gradient_n_sarsa(n, eps, q_hat, epsilon, gamma, env)
    env = gym.wrappers.RecordVideo(env, 'videos', name_prefix=name_prefix)
    s = env.reset()[0]
    # print(s)
    total_reward = 0
    pi = epsilon_greedy_policy(q_hat, 0, env.action_space.n)
    t=0

    while True:
      action_probs = pi(s)
      [a] = np.nonzero(action_probs)[0]
      next_state, reward, done, _, _ = env.step(a)
      total_reward += reward
      s = next_state

      if done:
          print(f"Total Reward: {total_reward}")
          print('Solved in {} steps'.format(t))
          print(next_state)
          break
      t+=1

    env.close()
    return scores

