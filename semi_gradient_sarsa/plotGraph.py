import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
import numpy as np
import pickle

file_name = 'scores.pkl'
with open(file_name, 'rb') as file:
    graph_scores = pickle.load(file)
sns.set()
plt.clf()
plt.plot(graph_scores)
plt.ylabel('Number of steps')
plt.xlabel('episodes')
plt.title('Acrobot - Semi Gradient n step Sarsa')

tick_interval_y = 500
plt.yticks(np.arange(min(graph_scores), max(graph_scores) + 1, tick_interval_y))

tick_interval_x = 30
plt.xticks(np.arange(0, 500, tick_interval_x))

plt.show()

# sns.set()
# plt.clf()
# plt.plot(graph_scores)
# plt.ylabel('Number of steps')
# plt.xlabel('episodes')
# plt.title('Acrobot - Sarsa')

# tick_interval_y = 200
# plt.yticks(np.arange(min(graph_scores), max(graph_scores) + 1, tick_interval_y))

# tick_interval_x = 50
# plt.xticks(np.arange(0, 500, tick_interval_x))

# plt.show()