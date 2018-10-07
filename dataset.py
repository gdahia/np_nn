import os
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


class _UnlabeledDataset:
  def __init__(self, images, names, shuffle):
    # initialize internal resources
    self.epochs = 0
    self._cursor = 0
    self._shuffle_flag = shuffle
    self._n_examples = len(images)
    self.input_shape = np.shape(images)

    # capture images and respective filenames
    self._images = images
    self._names = names

    # initial shuffle
    if shuffle:
      self._shuffle()

  def next_batch(self, size, incomplete=False):
    # sample data
    batch_images = self._images[self._cursor:self._cursor + size]
    batch_names = self._names[self._cursor:self._cursor + size]

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
        return batch_images, batch_names

      # update cursor
      rem = size - len(batch_images)
      self._cursor = rem

      # fill remainder of batch in next epoch
      rem_images = self._images[:rem]
      rem_names = self._names[:rem]
      batch_images = np.concatenate([batch_images, rem_images], axis=0)
      batch_names += rem_names

    return batch_images, batch_names

  def _shuffle(self):
    self._images = np.random.permutation(self._images)
    self._names = np.random.permutation(self._names)


class Handler:
  def __init__(self, path, split, shuffle=True):
    # load training data
    train_path = os.path.join(path, 'train')
    images, labels = utils.load_train_data(train_path)

    # dataset general info
    self.input_shape = np.shape(images)[1:]
    self.labels = {i: label for i, label in enumerate(np.unique(labels))}

    # convert labels to integers
    labels = utils.labels2int(labels)

    # split into train/val
    images, labels = utils.shuffle(images, labels)
    train, val = utils.split(images, labels, split)
    train_images, train_labels = train
    val_images, val_labels = val

    # load test data
    test_path = os.path.join(path, 'test')
    test_images, test_filenames = utils.load_test_data(test_path)

    # train dataset
    self.train = _LabeledDataset(train_images, train_labels, shuffle=shuffle)

    # validation dataset
    self.val = _LabeledDataset(val_images, val_labels, shuffle=False)

    # test dataset
    self.test = _UnlabeledDataset(test_images, test_filenames, shuffle=False)
