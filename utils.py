import os
import numpy as np
import urllib.request
import gzip
import struct


def download_mnist_file(path):
  url = os.path.join('http://yann.lecun.com/exdb/mnist', path)
  filename, _ = urllib.request.urlretrieve(url)
  return filename


def load_mnist_split(name):
  # get images
  images_filename = download_mnist_file('{}-images-idx3-ubyte.gz'.format(name))
  with gzip.open(images_filename, 'rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    nrows, ncols = struct.unpack(">II", f.read(8))
    data = np.frombuffer(f.read(), dtype=np.dtype(np.uint8).newbyteorder('>'))
    images = data.reshape((size, nrows, ncols))

  # get labels
  labels_filename = download_mnist_file('{}-labels-idx1-ubyte.gz'.format(name))
  with gzip.open(labels_filename, 'rb') as f:
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
