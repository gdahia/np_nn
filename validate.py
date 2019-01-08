import numpy as np


def accuracy(dataset, model, batch_size, preprocess):
  total = 0
  correct = 0
  current_epoch = dataset.epochs

  # iterate over dataset
  while dataset.epochs == current_epoch:
    # sample batch
    inputs, labels = dataset.next_batch(batch_size, incomplete=True)
    inputs = preprocess(inputs, inputs.shape[0])

    # predict outputs for inputs
    scores = model.infer(inputs)
    preds = np.argmax(scores, axis=-1)
    preds = np.reshape(preds, np.shape(labels))

    # update statistics
    total += len(inputs)
    correct += np.sum(preds == labels)

  return correct / total
