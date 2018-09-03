import numpy as np

import nn

from classifier.classifier import Classifier


class LogisticRegressor(Classifier):
  def __init__(self,
               units_ls,
               n_classes,
               input_dims,
               W_prior=None,
               b_prior=None,
               c_prior=None):
    # trick for using last units for shape
    prev_dims = input_dims

    # initialize variables
    self.Ws = []
    self.bs = []
    for units in units_ls:
      # initialize 'W's
      W_shape = (prev_dims, units)
      if W_prior is None:
        # xavier initialization
        val = 6 / np.sqrt(prev_dims + units)
        W = np.random.uniform(low=-val, high=val, size=W_shape)
      else:
        W = W_prior(W_shape)
      self.Ws.append(W)

      # initialize 'b's
      if b_prior is None:
        b = np.zeros(units)
      else:
        b = b_prior(units)
      self.bs.append(b)

      prev_dims = units

    # classification layer
    W_shape = (prev_dims, n_classes)
    if W_prior is None:
      # xavier initialization
      val = 6 / np.sqrt(prev_dims + n_classes)
      W = np.random.uniform(low=-val, high=val, size=W_shape)
    else:
      W = W_prior(W_shape)
    self.Ws.append(W)

    # initialize 'c'
    if c_prior is None:
      c = np.ones(n_classes, dtype=np.float32) / n_classes
    else:
      c = c_prior(n_classes)
    self.bs.append(c)

  def infer(self, x):
    x = np.reshape(x, (1, -1))
    for W, b in zip(self.Ws, self.bs):
      x = np.matmul(x, W) + b
    return nn.softmax(x)

  def _forward(self, inputs, labels):
    # flatten 'inputs'
    xs = np.reshape(inputs, (len(inputs), -1))

    # store every activation in forward pass
    activations = [xs]
    for W, b in zip(self.Ws, self.bs):
      xs = np.matmul(xs, W) + b
      activations.append(xs)

    # compute loss
    activations.append(nn.softmax(xs))
    loss = np.mean(
        nn.softmax_cross_entropy_with_logits(labels=labels, logits=xs))

    return activations, loss

  def _backward(self, activations, labels):
    # compute gradient wrt to softmax
    # TODO: derive if means can be computed when each gradient is ready to perform faster calculations
    # grad = np.mean(activations[-1] - labels, 0)
    grad = activations[-1] - labels

    # compute gradients with backpropagation
    grads = []
    for i in reversed(range(len(activations[:-2]))):
      # compute and store bias gradient
      db = np.mean(grad, 0)
      grads.append(db)

      # compute and store weights gradient
      dW = np.matmul(
          np.expand_dims(activations[i], -1), np.expand_dims(grad, 1))
      dW = np.mean(dW, 0)
      grads.append(dW)

      # continue backprop
      grad = np.matmul(grad, self.Ws[i].T)

    return grads

  def _update(self, grads, learning_rate):
    for i, grad in enumerate(reversed(grads)):
      if i % 2 == 0:
        self.Ws[i // 2] -= learning_rate * grad
      else:
        self.bs[i // 2] -= learning_rate * grad
