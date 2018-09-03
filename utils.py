import os
import numpy as np
import cv2


def load_train_data(path, image_dtype=np.float32):
  images = []
  labels = []
  for label in os.listdir(path):
    label_dir_path = os.path.join(path, label)
    for filename in os.listdir(label_dir_path):
      image_path = os.path.join(label_dir_path, filename)
      image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
      image = np.array(image, dtype=image_dtype) / 255

      images.append(image)
      labels.append(label)

  return np.array(images, dtype=image_dtype), labels


def load_test_data(path, image_dtype=np.float32):
  images = []
  for filename in os.listdir(path):
    image_path = os.path.join(path, filename)
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED) / 255
    image = np.array(image, dtype=image_dtype) / 255
    images.append(image)

  return np.array(images, dtype=image_dtype)


def shuffle(instances, labels):
  perm = np.random.permutation(len(instances))
  instances = np.array(instances)[perm]
  labels = np.array(labels)[perm]

  return instances, labels


def split(instances, labels, split):
  # compute split point
  split_point = np.round(len(instances) * split)

  # split instances
  train_instances = instances[:split_point]
  val_instances = instances[split_point:]

  # split labels
  train_labels = labels[:split_point]
  val_labels = labels[split_point:]

  return (train_instances, train_labels), (val_instances, val_labels)
