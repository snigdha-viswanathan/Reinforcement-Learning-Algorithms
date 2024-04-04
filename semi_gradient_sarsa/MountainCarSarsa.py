from nStepSarsa import train_sarsa
import pickle
# from nStepSarsaLinear import train_sarsa

import gym

env = gym.make('MountainCar-v0', render_mode='rgb_array')
name_prefix = 'MountainCarSarsa'
eps = 500
n = 3
epsilon = 0.1
gamma = 0.99
step_size = 0.05
tilings = 7
scores = train_sarsa(env, name_prefix, eps, n, epsilon, gamma, step_size, tilings)

file_name = 'scores.pkl'

# Writing the list to a file
with open(file_name, 'wb') as file:
    pickle.dump(scores, file)