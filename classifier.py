class Classifier:
  def __init__(self,
               units_ls,
               n_classes,
               W_prior=None,
               b_prior=None,
               logits_prior=None):
    pass

  def _forward(self, images, labels):
    pass

  def _backward(self):
    pass

  def _update(self):
    pass

  def train(self, images, labels, learning_rate):
    pass

  def infer(self, image):
    pass
