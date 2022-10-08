from builtins import range
from builtins import object
import numpy as np


from layers import *
from layer_utils import *
# from ..layers import *
# from ..layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
        # TwoLayerNet(X,input_dim=D, hidden_dim=H, num_classes=C, weight_scale=std)
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        #  affine forward
  
        # the intput dim of first layer is; input_dim = np.prod(input_shape)
        # the output dim of first layer is hidden_dim  
        self.params['W1'] = np.random.normal(0, weight_scale, (input_dim, hidden_dim))
        self.params['b1'] = np.zeros((hidden_dim))
        
        #  the intput dim of second layer is hidden_dim
        # the output dim of second layer is num_classes
        self.params['W2'] = np.random.normal(0, weight_scale, (hidden_dim, num_classes))
        self.params['b2'] = np.zeros((num_classes))
                      
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        # forward-pass of a 2-layer neural network:
        # input: X
        
        # first layer - affine and activation (relu)
        W1 = self.params['W1']
        b1 = self.params['b1']
        out1, cache1 = affine_relu_forward(X,W1 , b1)
        
        
        # second layer (it's the last layer without activation)
        # affine
        W2 = self.params['W2']
        b2 = self.params['b2']
        out2, cache2 = affine_forward(out1, W2, b2)
        scores = out2


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores
        
        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        # calculate loss  - according to the forward pass scores
        loss,dloss = softmax_loss(scores, y)
        loss += 0.5 * self.reg * np.sum(W1 * W1) + 0.5 * self.reg * np.sum(W2 * W2)
        
        
        # # backpropogation of a 2-layer neural network:
        # # second layer - recieve the gradiant with respect of x from the final output (dloss)
        dx2, dW2, db2 = affine_backward(dloss, cache2)
        # first layer - recieve the gradiant with respect of x of second layer (dx2)
        _, dW1, db1 = affine_relu_backward(dx2, cache1)
        # dx1 is unused
        
        dW1 += self.reg * W1
        dW2 += self.reg * W2

        grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}
      
      
      
      

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
if __name__ == "__main__":
  np.random.seed(231)
  N, D, H, C = 3, 5, 50, 7
  X = np.random.randn(N, D)
  y = np.random.randint(C, size=N)

  std = 1e-3
  model = TwoLayerNet(input_dim=D, hidden_dim=H, num_classes=C, weight_scale=std)

  print('Testing initialization ... ')
  W1_std = abs(model.params['W1'].std() - std)
  b1 = model.params['b1']
  W2_std = abs(model.params['W2'].std() - std)
  b2 = model.params['b2']
  assert W1_std < std / 10, 'First layer weights do not seem right'
  assert np.all(b1 == 0), 'First layer biases do not seem right'
  assert W2_std < std / 10, 'Second layer weights do not seem right'
  assert np.all(b2 == 0), 'Second layer biases do not seem right'

  print('Testing test-time forward pass ... ')
  model.params['W1'] = np.linspace(-0.7, 0.3, num=D*H).reshape(D, H)
  model.params['b1'] = np.linspace(-0.1, 0.9, num=H)
  model.params['W2'] = np.linspace(-0.3, 0.4, num=H*C).reshape(H, C)
  model.params['b2'] = np.linspace(-0.9, 0.1, num=C)
  X = np.linspace(-5.5, 4.5, num=N*D).reshape(D, N).T
  scores = model.loss(X)
  
  correct_scores = np.asarray(
    [[11.53165108,  12.2917344,   13.05181771,  13.81190102,  14.57198434, 15.33206765,  16.09215096],
    [12.05769098,  12.74614105,  13.43459113,  14.1230412,   14.81149128, 15.49994135,  16.18839143],
    [12.58373087,  13.20054771,  13.81736455,  14.43418138,  15.05099822, 15.66781506,  16.2846319 ]])
  
  scores_diff = np.abs(scores - correct_scores).sum()
  assert scores_diff < 1e-6, 'Problem with test-time forward pass'

  print('Testing training loss (no regularization)')
  y = np.asarray([0, 5, 1])
  loss, grads = model.loss(X, y)
  print(loss)
  correct_loss = 3.4702243556
  assert abs(loss - correct_loss) < 1e-10, 'Problem with training-time loss'
  
  model.reg = 1.0
  loss, grads = model.loss(X, y)
  correct_loss = 26.5948426952
  assert abs(loss - correct_loss) < 1e-10, 'Problem with regularization loss'

  # Errors should be around e-7 or less
  from gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
  for reg in [0.0, 0.7]:
    print('Running numeric gradient check with reg = ', reg)
    model.reg = reg
    loss, grads = model.loss(X, y)

    for name in sorted(grads):
      f = lambda _: model.loss(X, y)[0]
      grad_num = eval_numerical_gradient(f, model.params[name], verbose=False)
      print('%s relative error: %.2e' % (name, rel_error(grad_num, grads[name])))
  print(1)