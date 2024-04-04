import tileCoding
import numpy as np

class QHat_Function():
  def __init__(self, num_tilings, step_size, env):
    self.env = env
    self.num_tilings = num_tilings
    self.num_tiles = num_tilings
    self.max_size = 10000
    self.alpha = step_size / num_tilings
    self.iht = tileCoding.IHT(self.max_size)
    self.w = np.zeros(self.max_size)
    self.num_actions = env.action_space.n
    if self.env.spec.name == 'MountainCar':
      self.x_scale = self.num_tilings / (self.env.observation_space.high[0] - self.env.observation_space.low[0])
      self.v_scale = self.num_tilings / (self.env.observation_space.high[1] - self.env.observation_space.low[1])
    elif self.env.spec.name == 'Acrobot':
      self.scale0 = self.num_tilings / (self.env.observation_space.high[0] - self.env.observation_space.low[0])
      self.scale1 = self.num_tilings / (self.env.observation_space.high[1] - self.env.observation_space.low[1])
      self.scale2 = self.num_tilings / (self.env.observation_space.high[2] - self.env.observation_space.low[2])
      self.scale3 = self.num_tilings / (self.env.observation_space.high[3] - self.env.observation_space.low[3])
      self.scale4 = self.num_tilings / (self.env.observation_space.high[4] - self.env.observation_space.low[4])
      self.scale5 = self.num_tilings / (self.env.observation_space.high[5] - self.env.observation_space.low[5])
    else:
      self.scale0 = self.num_tilings / (self.env.observation_space.high[0] - self.env.observation_space.low[0])
      self.scale1 = self.num_tilings / (self.env.observation_space.high[1] - self.env.observation_space.low[1])
      self.scale2 = self.num_tilings / (self.env.observation_space.high[2] - self.env.observation_space.low[2])
      self.scale3 = self.num_tilings / (self.env.observation_space.high[3] - self.env.observation_space.low[3])
  def get_scaled_value(self, s, a):
    # print(s)
    if self.env.spec.name == 'MountainCar':
      scaled_state = [s[0] * self.x_scale, s[1] * self.v_scale]
    elif self.env.spec.name == 'Acrobot':
      scaled_state = [s[0] * self.scale0, s[1] * self.scale1, s[2] * self.scale2, s[3] * self.scale3, s[4] * self.scale4, s[5] * self.scale5]
    else:
      scaled_state = [s[0] * self.scale0, s[1] * self.scale1, s[2] * self.scale2, s[3] * self.scale3]
    scaled_val = tileCoding.tiles(self.iht, self.num_tilings, scaled_state, [a])
    # print(scaled_val)
    return scaled_val
    # scaled_state = [round(s) for s in scaled_state]
    # return scaled_state
  def predict(self, s, a=None):
    # print("In predict")
    if a is None:
      features = [self.get_scaled_value(s, a) for a in range(self.env.action_space.n)]
    else:
      features = [self.get_scaled_value(s, a)]
    predicted_val = [np.sum(self.w[f]) for f in features]
    return predicted_val
  def train(self, s, a, G):
    # print("In train, s:", s)
    scaled_val = self.get_scaled_value(s,a)
    predicted_val = np.sum(self.w[scaled_val])
    delta = G - predicted_val
    # print("G: ",G," pred val: ", predicted_val)
    self.w[scaled_val] += self.alpha * delta
  def reset_w(self):
    self.w = np.zeros(self.max_size)