import numpy as np

import nn.utils as nn

from nn.models.neural_net import NeuralNet


class Feedforward(NeuralNet):
  def __init__(self,
               W_ls,
               ops,
               activation_fns,
               loss_fn,
               optimizer,
               infer_fns=None,
               dropout_keep_probs=None):
    # assert number of layers and activation functions match
    if len(W_ls) != len(activation_fns):
      raise ValueError(
          'number of layers and activation functions do not match')

    # assert number of layers and ops match
    if len(W_ls) != len(ops):
      raise ValueError('number of layers and ops do not match')

    # assert every layer has dropout keep probability or none has
    if dropout_keep_probs is not None:
      if len(dropout_keep_probs) != len(W_ls):
        raise ValueError(
            'number of layers and dropout keep probabilities do not match')

    # capture activation functions and resp grads
    self._ops = ops
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

    # initialize variables and optimizers
    self._Ws = []
    self._W_optimizers = []
    self._bs = []
    self._b_optimizers = []
    for W in W_ls:
      # glorot-initialize layer weights
      fan_in, fan_out = nn.get_fans(W)
      val = np.sqrt(6 / (fan_in + fan_out))
      W = np.random.uniform(low=-val, high=val, size=W)
      self._W_optimizers.append(optimizer(W.shape))
      self._Ws.append(W)

      # zero initialize layer biases
      b = np.zeros(W.shape[-1])
      self._b_optimizers.append(optimizer(W.shape[-1]))
      self._bs.append(b)

  def infer(self, inputs):
    # group into layers
    W_ls = self._Ws
    bs = self._bs
    fns = self._infer_fns
    ops = self._ops
    layers = zip(W_ls, bs, fns, ops)

    # feedforward
    outputs = inputs
    for W, b, fn, op in layers:
      outputs = fn(op(outputs, W) + b)

    return outputs

  def forward(self, inputs, labels):
    W_ls = self._Ws
    bs = self._bs
    fns = self._activation_fns
    keep_probs = self._keep_probs
    ops = self._ops
    layers = zip(W_ls, bs, fns, keep_probs, ops)

    # store every activation and hidden output in forward pass
    hiddens = [inputs]
    activations = [None]
    for W, b, fn, keep_prob, op in layers:
      activation = op(hiddens[-1], W) + b
      activations.append(activation)

      hidden = fn(activation)
      hidden = nn.dropout(hidden, keep_prob)
      hiddens.append(hidden)

    # compute loss
    loss = np.mean(self._loss_fn(labels, hiddens[-1]))

    return activations, hiddens, loss

  def backward(self, activations, hiddens, labels):
    W_ls = self._Ws
    fns = self._activation_fns
    ops = self._ops

    # compute gradients with backpropagation
    grads = []
    grad = self._loss_fn.grad(labels, hiddens[-1])
    for i in range(len(activations) - 1, 0, -1):
      # compute and store bias gradient
      grad *= fns[i - 1].grad(activations[i])
      bgrad = grad
      if len(activations[i].shape) > 2:
        bgrad = np.sum(bgrad, axis=(1, 2))
      db = np.mean(bgrad, axis=0)

      # compute and store weights gradient
      dW = ops[i - 1].backprop_weights(
          inputs=hiddens[i - 1], weights=W_ls[i - 1], outputs_backprop=grad)
      dW = np.mean(dW, axis=0)

      # store gradients
      grads.append((dW, db))

      # continue backprop
      grad = ops[i - 1].backprop_inputs(
          inputs=hiddens[i - 1], weights=W_ls[i - 1], outputs_backprop=grad)

    return list(reversed(grads))

  def update(self, grads, **kwargs):
    for i, (dW, db) in enumerate(grads):
      self._Ws[i] -= self._W_optimizers[i].step(dW, **kwargs)
      self._bs[i] -= self._b_optimizers[i].step(db, **kwargs)
