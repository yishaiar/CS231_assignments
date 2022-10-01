from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W) # initialize the gradient as zero

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    

    # compute the loss and the gradient
    
    num_train = X.shape[0]
    
    for i in range(num_train):
        scores = X[i].dot(W) #SCORE VECTOR OF 10 CLASSES 
      
        # to avoid numric instabilty of dividing very large numbers shift scores value so highest number is 0 
        scores_exp = np.exp(scores - scores.max()) # numerically stable exponent vector
        # softmax for each score
        softmax = scores_exp / scores_exp.sum() 
        #  cross-entropy loss    
        loss -= np.log(softmax[y[i]])   

        # GRADIENT
        # http://intelligence.korea.ac.kr/jupyter/2020/06/30/softmax-classifer-cs231n.html
        #     
        
        softmax[y[i]] -= 1                  # update for gradient    
        dW += np.outer(X[i], softmax)       # gradient; x[i] is 3073 vector and softmax is 10 vector 
                                            # resulting with 10,3073 MATRIX 


    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += 2*reg *  W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    scores = X.dot(W) #mtrix; SCORE VECTOR OF 10 CLASSES for each x[i] sample by the number of X samples 
    scores =np.exp( scores - (np.max(scores,axis = 1)[:, np.newaxis])) # numerically stable exponent vector for all samples (matrix)
    # softmax for each score
    softmax = scores /(np.sum( scores, axis = 1)[:, np.newaxis])
    #  cross-entropy loss
 
    loss = -(np.log(softmax[range(num_train),y])).sum()
    # GRADIENT
    # http://intelligence.korea.ac.kr/jupyter/2020/06/30/softmax-classifer-cs231n.html
    #     
    softmax[range(num_train),y] -= 1                  # update for gradient 
    # softmax is a  matrix of num_samples x 10 classes  
    # X is a matrix of num_samples x 3073 featurs
    # when X transposed RESULT: ( 3073,num_samples,) DOT (num_samples x 10 classes) = ( 3073 x 10 classes) like dW
    dW += X.T.dot(softmax)       # gradient; x[i] is 3073 vector and softmax is 10 vector 
                                            # resulting with 10x3073 MATRIX 
    
  
      
        

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += 2*reg *  W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW





class Softmax(object):
  """ a Softmax classifier with softmax_loss_vectorized """

  def __init__(self):
    pass

  def train(self, X_train, y_train,lr, reg, num_iters=1000):
    """
    Train the classifier. 

    Inputs:
    - X: A numpy array of shape (num_train, D) containing the training data
      consisting of num_train samples each of dimension D.
    - y: A numpy array of shape (N,) containing the training labels, where
         y[i] is the label for X[i].
    """
    self.X_train = X
    self.y_train = y
    
  def predict(self, X, k=1, num_loops=0):
    """
    Predict labels for test data using this classifier.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data consisting
         of num_test samples each of dimension D.
    - k: The number of nearest neighbors that vote for the predicted labels.
    - num_loops: Determines which implementation to use to compute distances
      between training points and testing points.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    if num_loops == 0:
      dists = self.compute_distances_no_loops(X)
    elif num_loops == 1:
      dists = self.compute_distances_one_loop(X)
    elif num_loops == 2:
      dists = self.compute_distances_two_loops(X)
    else:
      raise ValueError('Invalid value %d for num_loops' % num_loops)

    return self.predict_labels(dists, k=k)








# if __name__ == "__main__":
#     X = np.arange(3072*50).reshape(50,3072)
#     X = np.hstack([X, np.ones((X.shape[0], 1))])
#     y = np.random.randint(10,size = 50)
#     print('X data shape: ', X.shape)
#     print('y data shape: ', y.shape)
#     W = np.random.randn(3073, 10) * 0.0001
#     loss, grad = softmax_loss_vectorized(W, X, y, 0.0)
#     print (loss)