import numpy as np


class conv2d:
  def __init__(self, strides, padding='valid'):
    if padding != 'valid' and padding != 'same':
      raise NotImplementedError('padding should be either "valid" or "same".')
    self._padding = padding

    if strides[0] != 1 or strides[-1] != 1:
      raise NotImplementedError('depth and channel strides should be 1.')
    self._strides = strides

  def __call__(self, inputs, filters):
    in_shape = np.shape(inputs)
    strides = self._strides

    if self._padding == 'same':
      # tf-like padding
      # heigh padding
      if (in_shape[1] % strides[1] == 0):
        pad_along_height = max(filters.shape[0] - strides[1], 0)
      else:
        pad_along_height = max(filters.shape[0] - (in_shape[1] % strides[1]),
                               0)

      # width padding
      if (in_shape[2] % strides[2] == 0):
        pad_along_width = max(filters.shape[1] - strides[2], 0)
      else:
        pad_along_width = max(filters.shape[1] - (in_shape[2] % strides[2]), 0)

      pad_top = pad_along_height // 2
      pad_bottom = pad_along_height - pad_top
      pad_left = pad_along_width // 2
      pad_right = pad_along_width - pad_left

      # pad input
      inputs = np.pad(inputs, ((0, 0), ((pad_top, pad_bottom)),
                               (pad_left, pad_right), (0, 0)), 'constant')

      # empty output
      out_height = (in_shape[1] + strides[1] - 1) // strides[1]
      out_width = (in_shape[2] + strides[2] - 1) // strides[2]
      out_shape = (in_shape[0], out_height, out_width, filters.shape[-1])
      outputs = np.empty(out_shape, dtype=inputs.dtype)
    else:
      # empty output
      out_height = (in_shape[1] - filters.shape[0] + strides[1]) // strides[1]
      out_width = (in_shape[2] - filters.shape[1] + strides[2]) // strides[2]
      out_shape = (in_shape[0], out_height, out_width, filters.shape[-1])
      outputs = np.empty(out_shape, dtype=inputs.dtype)

    # flatten filters for dot computation
    flat_filters = np.reshape(filters, (-1, filters.shape[-1]))

    # convolve
    for i in range(outputs.shape[1]):
      for j in range(outputs.shape[2]):
        receptive_field = inputs[:, strides[1] * i:strides[1] * i +
                                 filters.shape[0], strides[2] *
                                 j:strides[2] * j + filters.shape[1], :]
        receptive_field = np.reshape(receptive_field, (in_shape[0], -1))

        outputs[:, i, j, :] = np.dot(receptive_field, flat_filters)

    return outputs

  def grad(self, inputs, filters):
    #TODO
    pass


class _functor:
  def __new__(*args):
    return args[0]._fn(*args[1:])


class linear(_functor):
  @staticmethod
  def _fn(x):
    return x

  @staticmethod
  def grad(x):
    return 1


class relu(_functor):
  @staticmethod
  def _fn(x):
    return np.maximum(x, 0)

  @staticmethod
  def grad(x):
    return np.array(x > 0)


class sigmoid(_functor):
  @staticmethod
  def _fn(x):
    exp_x = np.exp(x)
    return np.maximum(x >= 0, 1 / (1 + np.exp(-x)), exp_x / (1 + exp_x))

  @staticmethod
  def grad(x):
    return sigmoid(x) * (1 - sigmoid(x))


class softmax(_functor):
  @staticmethod
  def _fn(x, axis=-1):
    x_ = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

  @staticmethod
  def grad(x, axis=-1):
    #TODO
    pass


class softmax_cross_entropy_with_logits(_functor):
  @staticmethod
  def _fn(labels, logits, axis=-1):
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


class sigmoid_cross_entropy_with_logits(_functor):
  @staticmethod
  def _fn(prob, logits, axis=-1):
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
  activations /= keep_prob

  return activations
