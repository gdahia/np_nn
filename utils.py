import os
import numpy as np
import urllib.request
import gzip
import struct


def download_mnist_file(path, save_path):
  print('Donwloading {}...'.format(path))
  url = os.path.join('http://yann.lecun.com/exdb/mnist', path)
  urllib.request.urlretrieve(url, save_path)
  print('Done')


def load_mnist_split(name):
  # download images, if not yet downloaded
  images_filename = '{}-images-idx3-ubyte.gz'.format(name)
  images_path = os.path.join('data', 'mnist', images_filename)
  if not os.path.exists(images_path):
    # create mnist data folder
    if not os.path.exists(os.path.join('data', 'mnist')):
      os.makedirs(os.path.join('data', 'mnist'))
    download_mnist_file(images_filename, images_path)

  # get images
  with gzip.open(images_path, 'rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    nrows, ncols = struct.unpack(">II", f.read(8))
    data = np.frombuffer(f.read(), dtype=np.dtype(np.uint8).newbyteorder('>'))
    images = data.reshape((size, nrows, ncols))

  # download labels, if not yet downloaded
  labels_filename = '{}-labels-idx1-ubyte.gz'.format(name)
  labels_path = os.path.join('data', 'mnist', labels_filename)
  if not os.path.exists(labels_path):
    download_mnist_file(labels_filename, labels_path)

  # get labels
  with gzip.open(labels_path, 'rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    data = np.frombuffer(f.read(), dtype=np.dtype(np.uint8).newbyteorder('>'))
    labels = data.reshape((size, ))

  return images, labels


def shuffle(instances, labels):
  perm = np.random.permutation(len(instances))
  instances = np.array(instances)[perm]
  labels = np.array(labels)[perm]

  return instances, labels


def split(instances, labels, split):
  # compute split point
  split_point = int(np.round(len(instances) * split))

  # split instances
  train_instances = instances[:split_point]
  val_instances = instances[split_point:]

  # split labels
  train_labels = labels[:split_point]
  val_labels = labels[split_point:]

  return (train_instances, train_labels), (val_instances, val_labels)


def has_mnist():
  return os.path.exists(os.path.join('data', 'mnist'))
