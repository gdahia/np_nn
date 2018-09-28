import numpy as np

import nn.utils as nn

from nn.models.neural_net import NeuralNet


class Feedforward(NeuralNet):
  def __init__(self,
               units_ls,
               activation_fns,
               input_dims,
               loss_fn,
               infer_fns=None,
               dropout_keep_probs=None,
               W_prior=None,
               b_prior=None):
    # assert number of layers and activation functions match
    if len(units_ls) != len(activation_fns):
      raise ValueError(
          'number of layers and activation functions do not match')

    # assert every layer has dropout keep probability or none has
    if dropout_keep_probs is not None:
      if len(dropout_keep_probs) != len(units_ls):
        raise ValueError(
            'number of layers and dropout keep probabilities do not match')

    # capture activation functions and resp grads
    self._activation_fns = activation_fns
    if infer_fns is None:
      self._infer_fns = activation_fns
    elif len(infer_fns) == len(activation_fns):
      self._infer_fns = infer_fns
    else:
      raise ValueError(('number of inference activation functions and number'
                        'of training activation functions do not match'))

    # capture loss function
    self._loss_fn = loss_fn

    # capture and fix for no dropout
    self._keep_probs = None
    if dropout_keep_probs is None:
      self._keep_probs = [1] * len(self._activation_fns)
    else:
      self._keep_probs = dropout_keep_probs + [1]

    # trick for using previous units for shape
    self._input_dims = input_dims
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

  def infer(self, inputs):
    # flatten 'input_'
    output = np.reshape(inputs, (-1, self._input_dims))

    for W, b, fn in zip(self._Ws, self._bs, self._infer_fns):
      output = fn(np.matmul(output, W) + b)

    return output

  def forward(self, inputs, labels):
    # flatten 'inputs'
    inputs = np.reshape(inputs, (-1, self._input_dims))

    # store every activation and hidden output in forward pass
    hiddens = [inputs]
    activations = [None]
    for W, b, fn, keep_prob in zip(self._Ws, self._bs, self._activation_fns,
                                   self._keep_probs):
      activation = np.matmul(hiddens[-1], W) + b
      hidden = fn(activation)
      hidden = nn.dropout(hidden, keep_prob)

      activations.append(activation)
      hiddens.append(hidden)

    # compute loss
    loss = np.mean(self._loss_fn(labels, hiddens[-1]))

    return activations, hiddens, loss

  def backward(self, activations, hiddens, labels):
    # compute gradients with backpropagation
    grads = []
    grad = self._loss_fn.grad(labels, hiddens[-1])
    for i in range(len(activations) - 1, 0, -1):
      # compute and store bias gradient
      grad *= self._activation_fns[i - 1].grad(activations[i])
      db = np.mean(grad, axis=0)

      # compute and store weights gradient
      dW = np.matmul(
          np.expand_dims(hiddens[i - 1], -1), np.expand_dims(grad, 1))
      dW = np.mean(dW, axis=0)

      # store gradients
      grads.append((dW, db))

      # continue backprop
      grad = np.matmul(grad, self._Ws[i - 1].T)

    return reversed(grads)

  def update(self, grads, learning_rate):
    for i, (dW, db) in enumerate(grads):
      self._Ws[i] -= learning_rate * dW
      self._bs[i] -= learning_rate * db
