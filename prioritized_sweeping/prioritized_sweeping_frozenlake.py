# -*- coding: utf-8 -*-
"""Prioritized sweeping-frozenlake.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/13N1pyj4oXq08pf8vMf4rlSgwsckDIFcL
"""

#from ctypes import sizeof
import gym
import numpy as np
from collections import defaultdict
from queue import PriorityQueue
# Create Acrobot environment
env = gym.make('FrozenLake-v1')
#env._max_episode_steps = 20000
# Constants
gamma = 0.9  # Discount factor
alpha = 0.1 # Learning rate
n = 20 # Number of updates per iteration
epsilon = 0.3
num_actions = env.action_space.n
theta = 0.0
num_states = env.observation_space.n
# Initialize Q-values, Model, and priority queue
# Q_values = np.zeros((num_states, num_actions))
Q_values = {}
for state in range(num_states):
  Q_values[state] = {}
  for action in range(num_actions):
    Q_values[state][action] = 0
#Q_values = np.zeros((num_states, num_actions))
#Q_values = defaultdict(lambda: np.zeros(num_actions))
model = {}
#model = defaultdict(lambda: [0, [0]*num_actions])
priority_queue = PriorityQueue()# Storing rewards and next states for each action
states_visited = {}
#priority_queue = defaultdict(float)
#print(priority_queue.get())

def get_action(state, Q):
    action = ""
    maximum_next_reward = float("-inf")

    if np.random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q_values[state])

    return action

def epsilon_decay(initial_epsilon, episode, decay_rate):
    min_epsilon = 0.1
    # epsilon = min_epsilon + (initial_epsilon - min_epsilon) / (1 + decay_rate * episode)
    epsilon = initial_epsilon / (1 + decay_rate * episode)
    return epsilon

# Prioritized sweeping algorithm using tile encoding
def prioritized_sweeping():
    epsilon = 0.3
    steps = [0 for i in range(5000)]
    t =0
    for run in range(1):
      steps_ep=0
      priority_queue = PriorityQueue()
      states_visited = {}  # nxtState -> list[(curState, Action)...]
      model = {}
      Q_values = {}
      for state in range(num_states):
        Q_values[state] = {}
        for action in range(num_actions):
          Q_values[state][action] = 0
      reward_tot = []
      for episode in range(5000):  # Perform a fixed number of iterations
          state = env.reset()
          #print(state)
          #print(state)  # Initial state
          # Loop through each state
          reward_ep = 0
          while True:
              # if(np.random.uniform(0,1)<epsilon):
              #   action = np.random.choice(len(Q_values[state]))
              # else:
              #   action = np.argmax(Q_values[state])
              #action = np.argmax(Q_values[state])
              action = choose_action(state, Q_values)
              #print(action)
              next_state, reward, done, _ = env.step(action)
              #print(done,reward,"-",next_state)
              # Update the model with the observed transition
              model[(state, action)] = (reward, next_state)
              # Calculate the temporal difference (TD) error

              q_max = np.max(list(Q_values[next_state].values()))
              TD_error = np.abs(reward + (gamma * q_max )- Q_values[state][action])
              #print(TD_error)
              # If the TD_error is non-zero, insert the state-action pair into the priority queue
              if TD_error > theta:
                  priority_queue.put((-TD_error, (state,action)))
              #print(priority_queue)
              # Perform prioritized updates
              if next_state not in states_visited.keys():
                states_visited[next_state] = [(state, action)]
              else:
                states_visited[next_state].append((state, action))
              state = next_state
              steps_ep += reward
              reward_ep = reward_ep + reward
              for _ in range(n):
                  if priority_queue.empty():
                    break
                  (s, a) = priority_queue.get()[1]
                  #print(s,a)
                # Update Q-values for the highest priority state-action pair
                  r, s_ = model[(s, a)]
                  #r, s_ = model[tuple(s + [a])]
                  #print(s)
                  # Update Q-values for the highest priority state-action pair
                  q_max = np.max(list(Q_values[s_].values()))
                  Q_values[s][a] += alpha * (r + gamma * q_max - Q_values[s][a])

                  # Remove the state-action pair from the priority queue
                  #del priority_queue[max_priority_item]
                  # Loop for all S', A' predicted to lead to S
                  if s in states_visited.keys():
                    s_a_list = states_visited[s]
                    for (prev_s, prev_a) in s_a_list:
                      prev_r = model[(prev_s,prev_a)][0]
                      prev_q_max = np.max(list(Q_values[s].values()))
                      prev_TD_error = np.abs(reward + (gamma * q_max )- Q_values[prev_s][prev_a])
                      if prev_TD_error > theta:
                        priority_queue.put((-prev_TD_error, (prev_s,prev_a)))
                  # for (ss, aa), (rr, _) in model.items():
                  #     # print("New tate:",ss)
                  #     #print(ss)
                  #     # print("Old state:",s)
                  #     if ss == s:
                  #     #state_n, action_n = state_action_pair[:-1], state_action_pair[-1]
                  #     #print(state_n)

                  #     #predicted_next_state = model[state_action_pair][1]
                  #     #print(predicted_next_state)
                  #     #if(tuple(state_n) == tuple(s)):
                  #       # predicted_reward = model[state_action_pair][0]
                  #       #print(predicted_next_state)
                  #       q_max = np.max(Q_values[s])
                  #       TD_error = np.abs(rr + gamma * q_max - Q_values[ss][aa])
                  #       #print(TD_error)
                  #       if TD_error > theta:
                  #           #print("Hi")
                  #           priority_queue.put((-TD_error, (ss,aa)))
                  #       # else:
                  #       #   print("Bye")
              if done:
                  #steps.append(t)
                  #print(t)
                  break
          #epsilon = epsilon_decay(epsilon, episode, 0.001)
          reward_tot.append(reward_ep)
          steps[episode] = steps[episode] + steps_ep    #state = next_state
          print(steps[episode])#print(Q_values)
    return Q_values, steps, reward_tot

# Run the prioritized sweeping algorithm with tile encoding
q_values, steps, reward_tot = prioritized_sweeping()
#print(q_values)

fin_steps_epi = [x/1 for x in steps]
print(max(fin_steps_epi))

import matplotlib.pyplot as plt

#y_values = reward_tot
x_values = [x for x in range(5000)]
y_values = fin_steps_epi

fig, ax = plt.subplots(figsize=(10, 5))

# Plotting the graph
plt.plot(x_values, y_values)

# Adding labels to the axes
plt.xlabel('Episodes')
plt.ylabel('Reward')

# Adding a title to the graph
plt.title('Frozen Lake')

# plt.xticks(range(round(min(steps_count)), round(max(steps_count))+1, 1000))

# Display the graph
plt.show()
# sns.set()
# plt.clf()
# plt.plot(fin_steps_epi)
# plt.ylabel('Number of steps')
# plt.xlabel('Rewards')
# plt.title('Froxzen lake - Priority Sweeping')

# tick_interval = 3000
# plt.yticks(np.arange(min(steps), max(steps) + 1, tick_interval))
# reg = LinearRegression().fit(np.arange(len(steps)).reshape(-1, 1), np.array(steps).reshape(-1, 1))
# y_pred = reg.predict(np.arange(len(steps)).reshape(-1, 1))
# plt.plot(y_pred)
# plt.show()

