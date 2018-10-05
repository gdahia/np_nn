import argparse

import utils
import dataset

FLAGS = None


def main():
  # download dataset if it does not exist
  if not utils.dataset_exists(FLAGS.dataset):
    print('Donwloading and extracting dataset {}...'.format(FLAGS.dataset))
    utils.download_and_extract_dataset(FLAGS.dataset)
    print('Done')

  # load dataset
  print('Loading dataset...')
  data = dataset.Handler(utils.dataset_path(FLAGS.dataset), 0.8)
  print('Done')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--dataset', required=True, type=int, help='index of dataset to use')
  parser.add_argument(
      '--model_path',
      required=True,
      type=str,
      help='path to trained model pickle file')
  parser.add_argument(
      '--results_path',
      required=True,
      type=str,
      help='path in which to save inferred labels')

  FLAGS = parser.parse_args()

  main()
