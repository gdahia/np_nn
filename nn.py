import numpy as np


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


def softmax_cross_entropy_with_logits(labels, logits):
  # force arguments into np.array format
  labels = np.array(labels)
  logits = np.array(logits)

  # TODO: compare numerically both forms of softmax using tf implementation
  # cross_entropy = -np.sum(labels * (logits - np.log(np.sum(np.exp(logits)))), axis=1)
  cross_entropy = -np.sum(labels * np.log(softmax(logits)), axis=1)


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
