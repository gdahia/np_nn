import numpy as np

import utils


class _LabeledDataset:
  def __init__(self, images, labels, shuffle):
    # initialize internal resources
    self.epochs = 0
    self._cursor = 0
    self._shuffle_flag = shuffle
    self._n_examples = len(images)

    # capture images and labels
    self._images = images
    self._labels = labels

    # initial shuffle
    if shuffle:
      self._shuffle()

  def next_batch(self, size, incomplete=False):
    # sample data
    batch_images = self._images[self._cursor:self._cursor + size]
    batch_labels = self._labels[self._cursor:self._cursor + size]

    if size + self._cursor < self._n_examples:
      self._cursor += size
    else:
      # epoch completed
      self.epochs += 1

      # shuffle
      if self._shuffle_flag:
        self._shuffle()

      # return incomplete batches
      if incomplete:
        self._cursor = 0
        return batch_images, batch_labels

      # update cursor
      rem = size - len(batch_images)
      self._cursor = rem

      # fill remainder of batch in next epoch
      rem_images = self._images[:rem]
      rem_labels = self._labels[:rem]
      batch_images = np.concatenate([batch_images, rem_images], axis=0)
      batch_labels = np.concatenate([batch_labels, rem_labels], axis=0)

    return batch_images, batch_labels

  def _shuffle(self):
    self._images, self._labels = utils.shuffle(self._images, self._labels)


class MNIST:
  def __init__(self, split, shuffle=True):
    # load training data
    images, labels = utils.load_mnist_split('train')
    images = images / 255.0

    # dataset general info
    self.input_shape = np.shape(images)[1:]
    self.labels = list(set(labels))

    # split into train/val
    images, labels = utils.shuffle(images, labels)
    train, val = utils.split(images, labels, split)
    train_images, train_labels = train
    val_images, val_labels = val

    # load test data
    test_images, test_labels = utils.load_mnist_split('t10k')
    test_images = test_images / 255.0

    # train dataset
    self.train = _LabeledDataset(train_images, train_labels, shuffle=shuffle)

    # validation dataset
    self.val = _LabeledDataset(val_images, val_labels, shuffle=False)

    # test dataset
    self.test = _LabeledDataset(test_images, test_labels, shuffle=False)
