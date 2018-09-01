import numpy as np


def softmax(x):
  x_ = x - np.max(x)
  exp_x = np.exp(x_)
  return exp_x / np.sum(exp_x)


def sigmoid(x):
  if x >= 0:
    return 1 / (1 + np.exp(-x))
  else:
    exp_x = np.exp(x)
    return exp_x / (1 + exp_x)
