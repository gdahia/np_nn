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
