# np_nn
A bare-bones, `numpy` only, toy neural network library.
Its only dependency is `numpy`, except for the CNN, which uses `cv2` to resize images.

It currently features implementations of:
* a linear logistic regression model, in `train.logistic_regression`;
* a ReLU feedforward netwrok, in `train.mlp`;
* a Convolutional neural network, in `train.cnn`.

All scripts use the [MNIST dataset](http://yann.lecun.com/exdb/mnist/), which is automatically downloaded if not yet in
the folder `data/mnist`.

## `nn.models.Feedforward`
The library is general in the sense that all the above models are objects of a single class, `nn.models.Feedforward`.
This class takes as argument a list of weight shapes
(and initializes them with [Glorot initialization](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)),
op functors as operations, activation functors as activation functions, and a loss functor; all of them from `nn.utils`.

## `nn.utils`
### Op functors
The op functors in `nn.utils` (at least those that are fully implemented) provide three methods. As an example,
consider the functor `nn.matmul`:
* `nn.matmul(inputs, weights)` computes the matrix product of `inputs` and weights;
* `nn.backprop_weights(inputs, weights, outputs_backprop)` backpropagates `outputs_backprop` through `weights`;
* `nn.backprop_inputs(inputs, weights, outputs_backprop)` backpropagates `outputs_backprop` through `inputs`.

The available op functors are `nn.matmul` and `nn.conv2d`.
Unfortunately, `nn.conv2d`'s backpropagation implementations are very time-consuming and do not yet work for 

### Activation and loss functors
The activation and loss functors in `nn.utils` provide two methods. As an example, consider the functor `nn.relu`:
* `nn.relu(x)` computes `np.maximum(x, 0)`;
* `nn.relu.grad(x)` computes `np.array(x > 0)`.

Other examples are
* `nn.softmax_cross_entropy_with_logits`, which could be implemented as a op functor to backpropagate through labels.
Using this, combined with last layer activation `nn.linear`, is currently the only way to train such a model,
as `nn.softmax.grad` is not yet implemented;
* `nn.linear`, required for models with no hidden non-linearities, like `logistic_regression`;
* `nn.sigmoid`.
