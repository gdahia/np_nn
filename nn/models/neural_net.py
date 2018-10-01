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
  def update(self, grads, optimizer):
    pass

  def train(self, inputs, labels, **kwargs):
    activations, hiddens, loss = self.forward(inputs, labels)
    grads = self.backward(activations, hiddens, labels)
    self.update(grads, **kwargs)

    return loss
