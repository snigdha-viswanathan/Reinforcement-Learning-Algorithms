from nStepSarsa import train_sarsa
import gym
import pickle

env = gym.make('Acrobot-v1', render_mode='rgb_array')
name_prefix = 'AcrobotSarsa'
# eps = 500
# n = 3
# epsilon = 0.1 / 0.01
# gamma = 0.99
# step_size = 0.08 / 0.04
# tilings = 4
eps = 500
n = 3
epsilon = 0.01
gamma = 0.99
step_size = 0.07
tilings = 4
scores = train_sarsa(env, name_prefix, eps, n, epsilon, gamma, step_size, tilings)

file_name = 'scores.pkl'

# Writing the list to a file
with open(file_name, 'wb') as file:
    pickle.dump(scores, file)