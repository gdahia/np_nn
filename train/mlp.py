import os
import argparse
import numpy as np
import copy

import utils
import nn
import dataset
import validate

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

  # create model
  print('Initializing model...')
  model = nn.models.Feedforward(
      units_ls=[512, 512, 512],
      activation_fns=[nn.relu] * 3,
      activation_dfns=[nn.drelu] * 3,
      n_classes=len(data.labels),
      input_dims=np.prod(data.input_shape))
  print('Done')

  # initial learning rate
  learning_rate = nn.linear_decay(FLAGS.learning_rate,
                                  FLAGS.learning_rate / 100, 5000)

  # early stopping vars
  best_model = None
  best_accuracy = 0
  faults = 0

  # train
  print('Training...')
  for step in range(1, FLAGS.steps + 1):
    # sample batch
    images, labels = data.train.next_batch(FLAGS.batch_size)
    labels = nn.one_hot(labels, depth=len(data.labels))

    # train step
    loss = model.train(images, labels, learning_rate(step))

    # print loss
    if step % FLAGS.loss_steps == 0:
      print('Step {}: loss = {}'.format(step, loss))

    # validate
    if step % FLAGS.val_steps == 0:
      accuracy = validate.accuracy(data.val, model, FLAGS.batch_size)
      print('Accuracy = {}'.format(accuracy))

      # early stopping
      if accuracy > best_accuracy:
        best_model = copy.deepcopy(model)
        # TODO: update best model saved copy here?
        best_accuracy = accuracy
        faults = 0
      else:
        faults += 1
        if faults >= FLAGS.tolerance:
          print('Training stopped early')
          break

  print('Done')
  print('Best accuracy = {}'.format(best_accuracy))

  # save model
  if FLAGS.save_path is not None:
    import pickle

    print('Saving model to {}...'.format(FLAGS.save_path))

    # create directory if does not exist
    dirname = os.path.dirname(FLAGS.save_path)
    if dirname:
      os.makedirs(dirname, exist_ok=True)

    with open(FLAGS.save_path, 'wb') as output:
      pickle.dump(best_model, output)

    print('Done')


if __name__ == '__main__':
  # parse arguments
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--dataset', required=True, type=int, help='index of dataset to use')
  parser.add_argument('--batch_size', default=8, type=int, help='batch size')
  parser.add_argument(
      '--learning_rate',
      default=1e-2,
      type=float,
      help='initial learning rate')
  parser.add_argument(
      '--steps', default=10000, type=int, help='training steps')
  parser.add_argument(
      '--loss_steps',
      default=10,
      type=int,
      help='interval between loss prints')
  parser.add_argument(
      '--val_steps',
      default=100,
      type=int,
      help='interval between validations')
  parser.add_argument(
      '--tolerance',
      default=5,
      type=int,
      help='maximum number of early stopping faults')
  parser.add_argument(
      '--save_path', type=str, help='path to save trained model')
  parser.add_argument('--seed', type=int, help='random seed')
  FLAGS = parser.parse_args()

  # set random seed
  np.random.seed(FLAGS.seed)

  main()
