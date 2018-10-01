from nn.optimizer.optimizer import Optimizer


class GradientDescent(Optimizer):
  def __init__(self, shape):
    pass

  def step(self, grad, **kwargs):
    return kwargs['learning_rate'] * grad
