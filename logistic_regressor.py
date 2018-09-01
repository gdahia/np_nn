import numpy as np

import nn

from classifier import Classifier


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
        W = np.random.normal(size=W_shape)
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
      W = np.random.normal(size=W_shape)
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

    # store every node in forward pass
    nodes = []
    for W, b in zip(self.Ws, self.bs):
      xs = np.matmul(xs, W) + b
      nodes.append(xs)

    # compute loss
    logits = xs
    labels = np.array(labels)
    # TODO: compare numerically both forms of softmax using tf implementation
    # loss = np.mean(-np.mean(labels * (logits - np.log(np.sum(np.exp(logits)))), axis=1))
    loss = np.mean(-np.mean(labels * np.log(nn.softmax(logits)), axis=1))
    nodes.append(loss)

    return nodes

  def _backward(self):
    pass

  def _update(self):
    pass
