import os
import argparse
import numpy as np
import pickle

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
  print('Loading dataset {}...'.format(FLAGS.dataset))
  data = dataset.Handler(utils.dataset_path(FLAGS.dataset), 0.8)
  print('Done')

  # load model
  print('Loading model {}...'.format(FLAGS.model_path))
  model = None
  with open(FLAGS.model_path, 'rb') as model_file:
    model = pickle.load(model_file)
  print('Done')

  # infer for every image
  print('Inferring...')
  preds = []
  names = []
  while data.test.epochs == 0:
    # sample batch
    batch_inputs, batch_names = data.test.next_batch(
        FLAGS.batch_size, incomplete=True)

    # infer for batch
    batch_outputs = model.infer(batch_inputs)
    batch_preds = np.argmax(batch_outputs, axis=-1)

    preds.extend(batch_preds)
    names.extend(batch_names)
  print('Done')

  # create results file directory tree
  dirname = os.path.dirname(FLAGS.results_path)
  dirname = os.path.abspath(dirname)
  if not os.path.exists(dirname):
    os.makedirs(dirname)

  # save results to file
  print('Saving results to {}...'.format(FLAGS.results_path))
  with open(FLAGS.results_path, 'w') as output:
    for name, pred in zip(names, preds):
      print(name, pred, file=output)
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
  parser.add_argument('--batch_size', default=128, type=int)

  FLAGS = parser.parse_args()

  main()
