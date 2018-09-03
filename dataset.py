import os
import numpy as np

import utils


class Dataset:
  def __init__(self, path, split, shuffle=True):
    # initialize internal resources
    self._epochs = 0
    self._cursor = 0
    self._shuffle = shuffle

    # load training data
    train_path = os.path.join(path, 'train')
    images, labels = utils.load_train_data(train_path)

    # split into train/val
    images, labels = utils.shuffle(images, labels)
    train, val = self._split(images, labels)
    self._train_images, self._train_labels = train
    self._val_images, self._val_labels = val

    # initial shuffle
    if shuffle:
      self._shuffle()

    # load test data
    test_path = os.path.join(path, 'test')
    self._test_images = utils.load_test_data(test_path)

  def next_batch(self, size, incomplete=False):
    # sample data
    batch_images = self._train_images[self._cursor:self._cursor + size]
    batch_labels = self._train_labels[self._cursor:self._cursor + size]

    # fill remainder of batch in next epoch
    if size + self._cursor > len(self._train_images):
      # shuffle
      if self._shuffle:
        self._shuffle()

      rem_images = self._train_images[:size - len(batch_images)]
      rem_labels = self._train_labels[:size - len(batch_labels)]
      batch_images = np.concat([batch_images, rem_images])
      batch_labels = np.concat([batch_labels, rem_labels])

    return batch_images, batch_labels

  def _shuffle(self):
    shuffled = utils.shuffle(self._train_images, self._train_labels)
    self._train_images, self._train_labels = shuffled
