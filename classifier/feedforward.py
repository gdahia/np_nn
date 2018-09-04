import numpy as np

import nn

from classifier.classifier import Classifier


class Feedforward(Classifier):
  def __init__(self,
               units_ls,
               activation_fns,
               activation_dfns,
               n_classes,
               input_dims,
               W_prior=None,
               b_prior=None,
               c_prior=None):
    # TODO: add docstring
    # capture activation functions and resp grads
    self._fns = activation_fns + [lambda x: x]
    self._dfns = activation_dfns + [lambda _: 1]

    # trick for using last units for shape
    prev_dims = input_dims

    # initialize variables
    self._Ws = []
    self._bs = []
    for units in units_ls:
      # initialize 'W's
      W_shape = (prev_dims, units)
      if W_prior is None:
        # xavier initialization
        val = 6 / np.sqrt(prev_dims + units)
        W = np.random.uniform(low=-val, high=val, size=W_shape)
      else:
        W = W_prior(W_shape)
      self._Ws.append(W)

      # initialize 'b's
      if b_prior is None:
        b = np.zeros(units)
      else:
        b = b_prior(units)
      self._bs.append(b)

      prev_dims = units

    # classification layer
    W_shape = (prev_dims, n_classes)
    if W_prior is None:
      # xavier initialization
      val = 6 / np.sqrt(prev_dims + n_classes)
      W = np.random.uniform(low=-val, high=val, size=W_shape)
    else:
      W = W_prior(W_shape)
    self._Ws.append(W)

    # initialize 'c'
    if c_prior is None:
      c = np.ones(n_classes, dtype=np.float32) / n_classes
    else:
      c = c_prior(n_classes)
    self._bs.append(c)

  def infer(self, x):
    # flatten 'x'
    x = np.reshape(x, (-1, self._Ws[0].shape[0]))

    # relu layers
    for W, b in zip(self._Ws[:-1], self._bs[:-1]):
      x = nn.relu(np.matmul(x, W) + b)

    # softmax layer
    x = np.matmul(x, self._Ws[-1]) + self._bs[-1]
    y = nn.softmax(x, axis=1)

    return y

  def _forward(self, inputs, labels):
    # flatten 'inputs'
    xs = np.reshape(inputs, (len(inputs), -1))

    # store every activation in forward pass
    activations = [xs]
    for W, b, fn in zip(self._Ws, self._bs, self._fns):
      xs = fn(np.matmul(xs, W) + b)
      activations.append(xs)

    # softmax predictions
    preds = nn.softmax(xs, axis=1)
    activations.append(preds)

    # compute loss
    loss = np.mean(nn.cross_entropy(probs=labels, preds=preds, axis=1))

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
      grad *= self._dfns[i](activations[i + 1])
      db = np.mean(grad, axis=0)
      grads.append(db)

      # compute and store weights gradient
      dW = np.matmul(
          np.expand_dims(activations[i], -1), np.expand_dims(grad, 1))
      dW = np.mean(dW, axis=0)
      grads.append(dW)

      # continue backprop
      grad = np.matmul(grad, self._Ws[i].T)

    return grads

  def _update(self, grads, learning_rate):
    for i, grad in enumerate(reversed(grads)):
      if i % 2 == 0:
        self._Ws[i // 2] -= learning_rate * grad
      else:
        self._bs[i // 2] -= learning_rate * grad
