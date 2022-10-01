from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    
    
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
                
                # condition for gradient is margin>0:
                # gradient of correct samples
                dW[:,y[i]] -= X[i]
                # gradient of incorrect samples
                dW[:,j] += X[i]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dW /= num_train   # scale gradient ovr the number of samples
    dW += 2 * reg * W # append partial derivative of regularization term

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # X[i] is a 3073 vector

    # X[i].dot(W) is 10 columns vector
    
    num_train = X.shape[0]
    # 500 samples with 10 classes:
    scores = X.dot(W) #10 columns and 500 rows 
    
    correct_class_score = scores[np.arange(num_train),y][:, np.newaxis] #vector with 500 cells 
    
    
    margin = scores - correct_class_score + 1  # note delta = 1
    
    # the margin at j ==y[i] and negative would not be added to loss
    margin[np.arange(num_train),y] = 0
    margin[margin<0] = 0
    loss = np.sum(margin[:])
    
    dW = (margin > 0).astype(int)    #  gradient with respect to Y_hat

    dW[np.arange(num_train), y] -= dW.sum(axis=1) # update gradient to include correct labels
    dW = X.T @ dW    # gradient with respect to W

    
    
    
    # ind = np.zeros(margin.shape)
    # # non empty index is true
    # ind[margin>0] = 1
    
    # # # column number
    # # j=0
    # # ind_1=ind[:,j]
    # # # expand to the dimension of x of 
    # # ind_1 = np.repeat(ind_1[:,np.newaxis], num_features, axis =1)
    # # sum over the number of x samples
    # # newX = np.sum(X*ind_1,axis=0)
    # # dW[:,j] += newX_1
    
    
    
    # ind =  np.repeat(ind[:,np.newaxis,:], num_features, axis = 1)
    # x_expand = np.repeat(X[:,:,np.newaxis],10,axis = 2)
    # # gradient
    # newX_2 = np.sum(x_expand*ind,axis=0)
    # # gradient of incorrect samples
    # dW += newX_2
    # # gradient of correct samples
    # newX_3 = np.sum(newX_2,axis=1)
    
    # # dW[:,y] -= newX
    
    # # number of times 
    # # sum_j = np.sum(ind,axis = 0)
    
    
    # # # condition for gradient is margin>0:
    # # # gradient of correct samples
    # # dW[:,y[i]] -= X[i]
    # # # gradient of incorrect samples
    # # dW[:,j] += X[i]

    # # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    loss /= num_train
    dW /= num_train


    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW



if __name__ == "__main__":
    X_dev = np.random.randint (128,size = (500,3072))
    X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])
    
    y_dev = np.random.randint (10,size = 500)
    # print(X_dev.shape)
    W = np.random.randn(3073, 10) * 0.0001 
    
    reg = 0.000005
    loss, grad = svm_loss_naive(W, X_dev, y_dev, reg)
    print(loss)
    loss, grad = svm_loss_vectorized(W, X_dev, y_dev, reg)
    print(loss)
    
    # print(W[1,:])