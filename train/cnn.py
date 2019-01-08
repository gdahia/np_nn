import os
import argparse
import numpy as np
import cv2
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
  kernels = [(3, 3, 1, 16), (3, 3, 16, 32), (3, 3, 32, len(data.labels))]
  model = nn.models.Feedforward(
      W_ls=kernels,
      ops=[nn.conv2d(strides=(1, 2, 2, 1), padding='valid')] * len(kernels),
      activation_fns=([nn.relu] * (len(kernels) - 1)) + [nn.linear],
      loss_fn=nn.softmax_cross_entropy_with_logits,
      optimizer=nn.optimizer.Momentum,
      infer_fns=([nn.relu] * (len(kernels) - 1)) + [nn.softmax])
  print('Done')

  # learning rate and momentum decay
  learning_rate = nn.linear_decay(FLAGS.learning_rate,
                                  FLAGS.learning_rate / 100, 5000)
  momentum = nn.linear_decay(FLAGS.momentum, 1.05 * FLAGS.momentum, 5000)

  # early stopping vars
  best_model = None
  best_accuracy = 0
  faults = 0

  def preprocess(inputs):
    inputs = [cv2.resize(input_, (17, 17)) for input_ in inputs]
    inputs = np.expand_dims(inputs, -1)
    return inputs

  # train
  print('Training...')
  for step in range(1, FLAGS.steps + 1):
    # sample batch
    images, labels = data.train.next_batch(FLAGS.batch_size)

    # adjust images, labels to net
    images = preprocess(images)
    labels = nn.one_hot(labels, depth=len(data.labels))

    # train step
    loss = model.train(
        images,
        labels,
        learning_rate=learning_rate(step),
        momentum=momentum(step))

    # print loss
    if step % FLAGS.loss_steps == 0:
      print('Step {}: loss = {}'.format(step, loss))

    # validate
    if step % FLAGS.val_steps == 0:
      accuracy = validate.accuracy(data.val, model, FLAGS.batch_size,
                                   preprocess)
      print('Accuracy = {}'.format(accuracy))

      # early stopping
      if accuracy > best_accuracy:
        best_model = copy.deepcopy(model)
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
      default=1e-3,
      type=float,
      help='initial learning rate')
  parser.add_argument('--momentum', default=0.9, type=float)
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
