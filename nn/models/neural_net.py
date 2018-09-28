from abc import ABC, abstractmethod


class NeuralNet(ABC):
  @abstractmethod
  def infer(self, inputs):
    pass

  @abstractmethod
  def forward(self, inputs, labels):
    pass

  @abstractmethod
  def backward(self, activations, hiddens, labels):
    pass

  @abstractmethod
  def update(self, grads, learning_rate):
    pass

  def train(self, inputs, labels, learning_rate):
    activations, hiddens, loss = self.forward(inputs, labels)
    grads = self.backward(activations, hiddens, labels)
    self.update(grads, learning_rate)

    return loss
