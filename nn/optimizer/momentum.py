import numpy as np

from nn.optimizer.optimizer import Optimizer


class Momentum(Optimizer):
  def __init__(self, shape):
    self._accumulation = np.zeros((shape), dtype=np.float32)

  def step(self, grad, **kwargs):
    self._accumulation = kwargs['momentum'] * self._accumulation + grad
    return kwargs['learning_rate'] * self._accumulation
