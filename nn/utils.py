import numpy as np


class matmul:
  def __new__(cls, inputs, weights):
    return np.matmul(inputs, weights)

  @staticmethod
  def backprop_weights(inputs, weights, outputs_backprop):
    inputs = np.expand_dims(inputs, -1)
    outputs_backprop = np.expand_dims(outputs_backprop, 1)
    return np.matmul(inputs, outputs_backprop)

  @staticmethod
  def backprop_inputs(inputs, weights, outputs_backprop):
    return np.matmul(outputs_backprop, weights.T)


class conv2d:
  def __init__(self, strides, padding='valid'):
    padding = padding.lower()
    if padding != 'valid' and padding != 'same':
      raise NotImplementedError('padding should be either "valid" or "same".')
    self._padding = padding

    if strides[0] != 1 or strides[-1] != 1:
      raise NotImplementedError('depth and channel strides should be 1.')
    self._strides = strides

  def __call__(self, inputs, weights):
    filters = weights
    in_shape = np.shape(inputs)
    strides = self._strides

    if self._padding == 'same':
      inputs = same_padding(inputs, strides, filters.shape)

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

    # convolve
    flat_filters = np.reshape(filters, (-1, filters.shape[-1]))
    for i in range(outputs.shape[1]):
      for j in range(outputs.shape[2]):
        rec_field = inputs[:, strides[1] * i:strides[1] * i + filters.shape[0],
                           strides[2] * j:strides[2] * j + filters.shape[1], :]
        rec_field = np.reshape(rec_field, (in_shape[0], -1))

        outputs[:, i, j, :] = np.dot(rec_field, flat_filters)

    return outputs

  def backprop_weights(self, inputs, weights, outputs_backprop):

    filters = weights
    in_shape = inputs.shape
    strides = self._strides

    # pad, if necessary
    if self._padding == 'same':
      inputs = same_padding(inputs, strides, filters.shape)

    # prepare indices and outputs_backprop for efficient computation
    k = [strides[i] * np.arange(outputs_backprop.shape[i]) for i in (1, 2)]
    k[0] = np.repeat(k[0], outputs_backprop.shape[2])
    k[1] = np.tile(k[1], outputs_backprop.shape[1])
    outputs_backprop = np.reshape(outputs_backprop,
                                  (in_shape[0], -1, filters.shape[3]))

    # compute gradient
    grad = np.empty((inputs.shape[0], ) + filters.shape, dtype=filters.dtype)
    for i in range(grad.shape[1]):
      for j in range(grad.shape[2]):
        for m in range(grad.shape[3]):
          rec_field = inputs[:, i + k[0], j + k[1], m]
          rec_field = np.reshape(rec_field, (in_shape[0], -1, 1))
          grad[:, i, j, m, :] = np.sum(rec_field * outputs_backprop, axis=1)
    grad *= in_shape[0]

    return grad

  def backprop_inputs(self, inputs, weights, outputs_backprop):
    filters = weights
    in_shape = inputs.shape
    strides = np.array(self._strides[1:3])

    # TODO: make it faster with indexing somehow
    # TODO: fix for 'SAME' padding. what prob is wrong is that
    # we are considering padding as valid input area, when it
    # is not. this means that these regions should not be propagated
    # and their outputs should not count as normal ones
    grad = np.zeros_like(inputs)
    for j in range(in_shape[0]):
      for c in np.ndindex(outputs_backprop.shape[1:3]):
        for m in range(in_shape[3]):
          for k in np.ndindex(filters.shape[:2]):
            index = c * strides + k
            if np.all(0 <= index) and np.all(index < grad.shape[1:3]):
              for i in range(filters.shape[3]):
                grad[j, index[0], index[1],
                     m] += filters[k[0], k[1], m,
                                   i] * outputs_backprop[j, c[0], c[1], i]

    return grad


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
    # reshape logits to match labels format
    logits = np.reshape(logits, labels.shape)

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
    # reshape logits to match labels format
    logits_shape = logits.shape
    logits = np.reshape(logits, labels.shape)

    # compute gradient
    grad = softmax(logits) - labels

    # reshape to logits shape
    grad = np.reshape(grad, logits_shape)

    return grad


class sigmoid_cross_entropy_with_logits:
  def __new__(cls, prob, logits, axis=-1):
    # use tf implementation, as described in:
    # https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
    return np.max(logits, 0) - logits * prob + np.log1p(
        np.exp(-np.abs(logits)))

  @staticmethod
  def grad(prob, logits):
    return sigmoid(logits) - prob


def cross_entropy(probs, preds, axis=-1):
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


def get_fans(shape):
  fan_in = shape[0] if len(shape) == 2 else np.prod(shape[1:])
  fan_out = shape[1] if len(shape) == 2 else shape[0]
  return fan_in, fan_out


def same_padding(inputs, strides, filters_shape):
  in_shape = inputs.shape

  # tf-like padding
  # height padding
  if (in_shape[1] % strides[1] == 0):
    pad_along_height = max(filters_shape[0] - strides[1], 0)
  else:
    pad_along_height = max(filters_shape[0] - (in_shape[1] % strides[1]), 0)

  # width padding
  if (in_shape[2] % strides[2] == 0):
    pad_along_width = max(filters_shape[1] - strides[2], 0)
  else:
    pad_along_width = max(filters_shape[1] - (in_shape[2] % strides[2]), 0)

  pad_top = pad_along_height // 2
  pad_bottom = pad_along_height - pad_top
  pad_left = pad_along_width // 2
  pad_right = pad_along_width - pad_left

  # pad input
  inputs = np.pad(inputs, ((0, 0), ((pad_top, pad_bottom)),
                           (pad_left, pad_right), (0, 0)), 'constant')

  return inputs
