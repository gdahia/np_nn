import argparse
import numpy as np

import utils

FLAGS = None


def main():
  # download dataset if it does not exist
  if not utils.dataset_exists(FLAGS.dataset):
    print('Donwloading and extracting dataset {}...'.format(FLAGS.dataset))
    utils.download_and_extract_dataset(FLAGS.dataset)
    print('Done')


if __name__ == '__main__':
  # parse arguments
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--dataset', required=True, type=int, help='index of dataset to use')
  parser.add_argument('--seed', type=int, help='random seed')
  FLAGS = parser.parse_args()

  # set random seed
  np.random.seed(FLAGS.seed)

  main()
