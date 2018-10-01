from abc import ABC, abstractmethod


class Optimizer(ABC):
  @abstractmethod
  def step(self, grad, **kwargs):
    pass
