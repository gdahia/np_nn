import argparse
import numpy as np

import utils
import nn
import classifier
import dataset

FLAGS = None


def main():
  # download dataset if it does not exist
  if not utils.dataset_exists(FLAGS.dataset):
    print('Donwloading and extracting dataset {}...'.format(FLAGS.dataset))
    utils.download_and_extract_dataset(FLAGS.dataset)
    print('Done')

  # load dataset
  data = dataset.Handler(utils.dataset_path(FLAGS.dataset), 0.8)

  # create classifier
  # TODO: take classifier as command line argument
  model = classifier.LinearSoftmax([], len(data.labels),
                                   np.prod(data.input_shape))

  # starting learning rate
  # TODO: when changing to decayed learning rate, update help
  learning_rate = FLAGS.learning_rate

  # train
  for step in range(1, FLAGS.steps + 1):
    # sample batch
    images, labels = data.train.next_batch(FLAGS.batch_size)
    labels = nn.one_hot(labels, depth=len(data.labels))

    # train step
    loss = model.train(images, labels, learning_rate)

    # print loss
    if step % FLAGS.loss_steps == 0:
      print('Step {}: loss = {}'.format(step, loss))


if __name__ == '__main__':
  # parse arguments
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--dataset', required=True, type=int, help='index of dataset to use')
  parser.add_argument('--batch_size', default=32, type=int, help='batch size')
  parser.add_argument(
      '--learning_rate', default=1e-2, type=float, help='learning rate')
  parser.add_argument(
      '--steps', default=10000, type=int, help='training steps')
  parser.add_argument(
      '--loss_steps',
      default=10,
      type=int,
      help='interval between loss prints')
  parser.add_argument('--seed', type=int, help='random seed')
  FLAGS = parser.parse_args()

  # set random seed
  np.random.seed(FLAGS.seed)

  main()
