import numpy as np


def relu(x):
  return np.maximum(x, 0)


def drelu(x):
  return (x > 0)


def softmax(x, axis=None):
  x_ = x - np.max(x, axis=axis, keepdims=True)
  exp_x = np.exp(x_)
  return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def sigmoid(x):
  if x >= 0:
    return 1 / (1 + np.exp(-x))
  else:
    exp_x = np.exp(x)
    return exp_x / (1 + exp_x)


def softmax_cross_entropy_with_logits(labels, logits, axis=None):
  # force arguments into np.array format
  labels = np.array(labels)
  logits = np.array(logits)

  # compute like tensorflow implementation:
  # https://github.com/tensorflow/tensorflow/blob/3e00e39c17124c9945fbad04a551a39ab4144935/tensorflow/core/kernels/xent_op.h#L110-L111
  cross_entropy = -np.sum(
      labels *
      (logits - np.log(np.sum(np.exp(logits), axis=axis, keepdims=True))),
      axis=axis,
      keepdims=True)

  return cross_entropy


def cross_entropy(probs, preds, axis=None):
  # make arguments be np arrays
  probs = np.array(probs)
  preds = np.array(preds)

  # compute cross entropy
  cross_entropy = -np.sum(probs * np.log(preds), axis=axis, keepdims=True)

  return cross_entropy


def one_hot(indices, depth, dtype=np.int32):
  onehot = np.zeros((len(indices), depth), dtype=dtype)
  for i, index in enumerate(indices):
    onehot[i, index] = 1
  return onehot


def linear_decay(initial, final, steps):
  def current(k):
    if k < steps:
      alpha = k / steps
      return (1 - alpha) * initial + alpha * final
    return final

  return current


def dropout(activations, keep_prob):
  # sample mask
  premask = np.random.uniform(low=0, high=1, size=np.shape(activations))
  mask = premask < keep_prob

  # drop units
  activations[~mask] = 0

  # adjust for weight scaling rule only during training
  activations *= keep_prob

  return activations
