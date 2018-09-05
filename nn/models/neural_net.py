from abc import ABC, abstractmethod


class NeuralNet(ABC):
  @abstractmethod
  def infer(self, x):
    pass

  @abstractmethod
  def _forward(self, inputs, labels):
    pass

  @abstractmethod
  def _backward(self, activations, labels):
    pass

  @abstractmethod
  def _update(self, grads, learning_rate):
    pass

  def train(self, inputs, labels, learning_rate):
    activations, loss = self._forward(inputs, labels)
    grads = self._backward(activations, labels)
    self._update(grads, learning_rate)

    return loss
