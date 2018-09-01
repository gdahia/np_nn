from abc import ABC, abstractmethod


class Classifier(ABC):
  @abstractmethod
  def infer(self, x):
    pass

  @abstractmethod
  def _forward(self, images, labels):
    pass

  @abstractmethod
  def _backward(self):
    pass

  @abstractmethod
  def _update(self):
    pass

  def train(self, images, labels, learning_rate):
    pass
