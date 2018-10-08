import numpy as np


class linear:
  def __new__(cls, x):
    return x

  @staticmethod
  def grad(x):
    return 1


class relu:
  def __new__(cls, x):
    return np.maximum(x, 0)

  @staticmethod
  def grad(x):
    return np.array(x > 0)


class sigmoid:
  def __new__(cls, x):
    exp_x = np.exp(x)
    return np.maximum(x >= 0, 1 / (1 + np.exp(-x)), exp_x / (1 + exp_x))

  @staticmethod
  def grad(x):
    return sigmoid(x) * (1 - sigmoid(x))


class softmax:
  def __new__(cls, x, axis=-1):
    x_ = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

  @staticmethod
  def grad(x, axis=-1):
    raise NotImplementedError()


class softmax_cross_entropy_with_logits:
  def __new__(cls, labels, logits, axis=-1):
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

  @staticmethod
  def grad(labels, logits):
    return softmax(logits) - labels


class sigmoid_cross_entropy_with_logits:
  def __new__(cls, prob, logits, axis=-1):
    cross_entropy = -(prob * np.log(sigmoid(logits)) +
                      (1 - prob) * np.log(1 - sigmoid(logits)))

    return cross_entropy

  @staticmethod
  def grad(prob, logits):
    return sigmoid(logits) - prob


def cross_entropy(probs, preds, axis=-1):
  # make arguments be np arrays
  probs = np.array(probs)
  preds = np.array(preds)

  # compute cross entropy
  cross_entropy = -np.sum(probs * np.log(preds), axis=axis, keepdims=True)

  return cross_entropy


def one_hot(indices, depth, dtype=np.float32):
  onehot = np.zeros((len(indices), depth), dtype=dtype)
  onehot[np.arange(len(indices)), indices] = 1
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
  activations /= keep_prob

  return activations
