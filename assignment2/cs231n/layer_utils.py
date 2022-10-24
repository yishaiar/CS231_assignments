# from layers import *
# from fast_layers import *

from cs231n.layers import *
from cs231n.fast_layers import *

def affine_relu_forward(x, w, b):
    """
    Convenience layer that perorms an affine transform followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b)
    # fc_cache is the affine_forward cached inputs; cache = (x, w, b) 
    # a is the out result; out = X@w+b
    

    out, relu_cache = relu_forward(a)
    # relu_cache is the relu_forward cached input; cache = a 
    # out is the relu result; out = np.maximum(0,a)
    
    
    cache = (fc_cache, relu_cache)
    # fc_cache is the affine_forward cached inputs; cache = (x, w, b) 
    # relu_cache is the relu_forward cached input; cache = a 

    return out, cache

def affine_relu_backward(dout, cache):
    """Backward pass for the affine-relu convenience layer.
    """
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db

# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

def affine_norm_activation_forward(x, w, b, bn_param,norm, activation,beta = None ,gamma = None):
    
    """
    Convenience layer that perorms an affine transform 
    followed by
    normalization: according to user choice - None/ batch norm / layer norm
    followed by
    activation: according to user choice - ReLU
    i.e affine - [None/batch/layer norm] - relu

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    A, fc_cache = affine_forward(x, w, b)
    # fc_cache is the affine_forward cached inputs; cache = (x, w, b) 
    # A is the out result; out = X@w+b
    
    if norm == 'batchnorm':
        B, norm_cache = batchnorm_forward(A, gamma, beta, bn_param)
    elif norm == 'layernorm':
        B, norm_cache = layernorm_forward(A, gamma, beta, bn_param)
    else: #if norm == None:
        B = A

    if activation == 'ReLU':
        out, relu_cache = relu_forward(B)
        # relu_cache is the relu_forward cached input; cache = A
        # out is the relu result; out = np.maximum(0,A)
    # else: other activation function
    
    # fc_cache is the affine_forward cached inputs; cache = (x, w, b) 
    # norm_cache TBD
    # relu_cache is the relu_forward cached input; cache = A 
    

    if norm != None:
        cache = (fc_cache, norm_cache,relu_cache)
        return out, cache
    # else:
    cache = (fc_cache ,relu_cache)
    return out, cache




def affine_norm_activation_backward(dout, cache,norm = None, activation = 'ReLU'):
    # def affine_norm_activation_forward(x, w, b, bn_param,norm = None, activation = 'ReLU'):

    # Backward pass for the affine-norm - activation layer.
    
    # get cache:
    if norm != None:
        fc_cache, norm_cache,relu_cache = cache
    else:
        fc_cache, relu_cache = cache
    
    
    if activation == 'ReLU':
        dB = relu_backward(dout, relu_cache)
    # else: other activation function

    if norm == 'batchnorm':
        dA, dgamma, dbeta = batchnorm_backward(dB, norm_cache)
    elif norm == 'layernorm':
        dA, dgamma, dbeta = layernorm_backward(dB, norm_cache)
    else: #if norm == None:
        dA = dB
    
    dx, dw, db = affine_backward(dA, fc_cache)
    return dx, dw, db

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

def conv_relu_forward(x, w, b, conv_param):
    """A convenience layer that performs a convolution followed by a ReLU.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    out, relu_cache = relu_forward(a)
    cache = (conv_cache, relu_cache)
    return out, cache


def conv_relu_backward(dout, cache):
    """Backward pass for the conv-relu convenience layer.
    """
    conv_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db


def conv_bn_relu_forward(x, w, b, gamma, beta, conv_param, bn_param):
    """Convenience layer that performs a convolution, a batch normalization, and a ReLU.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    - pool_param: Parameters for the pooling layer
    - gamma, beta: Arrays of shape (D2,) and (D2,) giving scale and shift
      parameters for batch normalization.
    - bn_param: Dictionary of parameters for batch normalization.

    Returns a tuple of:
    - out: Output from the pooling layer
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    an, bn_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(an)
    cache = (conv_cache, bn_cache, relu_cache)
    return out, cache


def conv_bn_relu_backward(dout, cache):
    """Backward pass for the conv-bn-relu convenience layer.
    """
    conv_cache, bn_cache, relu_cache = cache
    dan = relu_backward(dout, relu_cache)
    da, dgamma, dbeta = spatial_batchnorm_backward(dan, bn_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db, dgamma, dbeta


def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
    """Convenience layer that performs a convolution, a ReLU, and a pool.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    - pool_param: Parameters for the pooling layer

    Returns a tuple of:
    - out: Output from the pooling layer
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    s, relu_cache = relu_forward(a)
    out, pool_cache = max_pool_forward_fast(s, pool_param)
    cache = (conv_cache, relu_cache, pool_cache)
    return out, cache


def conv_relu_pool_backward(dout, cache):
    """Backward pass for the conv-relu-pool convenience layer.
    """
    conv_cache, relu_cache, pool_cache = cache
    ds = max_pool_backward_fast(dout, pool_cache)
    da = relu_backward(ds, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db
