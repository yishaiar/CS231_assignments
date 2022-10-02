from builtins import range
import numpy as np



def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k);(N,D)=(2,4*5*6)=(2,120)
    - w: A numpy array of weights, of shape (D, M)=(120,3)
    - b: A numpy array of biases, of shape (M,) = (3,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # num_minibatch,i,j,k = x.shape
    # size_minibatch = i*j*k
    # size_final = b.shape[0]
    # # scores  = np.zeros((num_minibatch,size_final))
    # scores = []
    # for X in x[ range(num_minibatch)]:
    #   X = X.reshape(size_minibatch)
    #   scores.append(  X@w+b)
    # out = np.array(scores)
    # print(out)
    
    
    # x_shape = x.shape   
    num_minibatch,size_minibatch = x.shape[0],np.prod(x.shape[1:])
    
    # reshape x to multiple minibatch vectors
    X = x.reshape(num_minibatch,size_minibatch)
    # final SCORES
    out = X@w+b
    # print(out)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k) = (10,2,3)
      - w: Weights, of shape (D, M) =(6,5)
      - b: Biases, of shape (M,) =(5,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)


    # forward pass
    W = np.random.randn(6, 5)
    X = np.random.randn(10, 6)
    (dout) D = W.dot(X) + b ; (10,5)

    # now suppose we had the gradient on D from above in the circuit
    dD = np.random.randn(*D.shape) # same shape as D (10,5)
    dW = X.T.dot(dD) = (6,10)*(10,5)= (6,5)
    dX = dD.dot(W.T) = (10,5)* (6, 5)* = (10,6)
    db = (1).dot(dD)] = (1,10)*(10,5) = (1,5)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    num_minibatches,size_minibatch = x.shape[0],np.prod(x.shape[1:])
    num_classes = w.shape[1]
    # reshape x to multiple minibatch vectors
    X = x.reshape(num_minibatches,size_minibatch)

    # dD = np.random.randn(dout.shape)
    dw = (X.T)@dout #.T gives the transpose of the matrix
    dX = dout@(w.T)
    ones = np.ones((1,num_minibatches))
    db = ones@dout
    # print(db.shape)

    # db = np.sum()
    dx = dX.reshape(x.shape)


  
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # reshape of x is un-necessary
    out = np.maximum(0,x)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dx = np.zeros_like(x)
    dx[x>0] = 1
    dx *= dout
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None
    ###########################################################################
    # TODO: Implement loss and gradient for multiclass SVM classification.    #
    # This will be similar to the svm loss vectorized implementation in       #
    # cs231n/classifiers/linear_svm.py.                                       #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # X training dataset not recieved
    # y is the correct class for each X
    # input x is the previously calculated *scores* f(w,X)= X.dot(W)
    # num_train,num_classes = x.shape
    num_train,_ = x.shape

    correct_class_score = x[np.arange(num_train),y][:, np.newaxis] #vector with 500 cells 
    
    
    margin = x - correct_class_score + 1  # note delta = 1
    
    # the margin at j ==y[i] and negative would not be added to loss
    margin[np.arange(num_train),y] = 0
    margin[margin<0] = 0
    
    loss = np.sum(margin[:])
    dx = (margin > 0).astype(float)    #  gradient with respect to x (scores), type int to avoid type bool

    dx[np.arange(num_train), y] -= dx.sum(axis=1) # update gradient to include correct labels
    
    
    dx /= num_train
    loss /= num_train

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None
    ###########################################################################
    # TODO: Implement the loss and gradient for softmax classification. This  #
    # will be similar to the softmax loss vectorized implementation in        #
    # cs231n/classifiers/softmax.py.                                          #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx




if __name__ == "__main__":
  # affine forward
  # num_inputs = 2
  # input_shape = (4, 5, 6)
  # output_dim = 3

  # input_size = num_inputs * np.prod(input_shape)
  # weight_size = output_dim * np.prod(input_shape)

  # x = np.linspace(-0.1, 0.5, num=input_size).reshape(num_inputs, *input_shape)
  # w = np.linspace(-0.2, 0.3, num=weight_size).reshape(np.prod(input_shape), output_dim)
  # b = np.linspace(-0.3, 0.1, num=output_dim)

  # out, _ = affine_forward(x, w, b)


# # affine  backward function
#   from gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
#   np.random.seed(231)
#   x = np.random.randn(10, 2, 3)
#   w = np.random.randn(6, 5)
#   b = np.random.randn(5)
#   dout = np.random.randn(10, 5)

#   dx_num = eval_numerical_gradient_array(lambda x: affine_forward(x, w, b)[0], x, dout)
#   dw_num = eval_numerical_gradient_array(lambda w: affine_forward(x, w, b)[0], w, dout)
#   db_num = eval_numerical_gradient_array(lambda b: affine_forward(x, w, b)[0], b, dout)

#   _, cache = affine_forward(x, w, b)
#   dx, dw, db = affine_backward(dout, cache)
  
#   
  # # relu backward
  # from gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
  # np.random.seed(231)
  # x = np.random.randn(10, 10)
  # dout = np.random.randn(*x.shape)

  # dx_num = eval_numerical_gradient_array(lambda x: relu_forward(x)[0], x, dout)

  # _, cache = relu_forward(x)
  # dx = relu_backward(dout, cache)

   # svm_loss
    np.random.seed(231)
    num_classes, num_inputs = 10, 50
    x = 0.001 * np.random.randn(num_inputs, num_classes)
    y = np.random.randint(num_classes, size=num_inputs)
    from gradient_check import eval_numerical_gradient, eval_numerical_gradient_array

    dx_num = eval_numerical_gradient(lambda x: svm_loss(x, y)[0], x, verbose=False)
    loss, dx = svm_loss(x, y)

 