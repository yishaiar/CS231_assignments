from .layers import *
# from layers import *


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
    """
    Backward pass for the affine-relu convenience layer
    """
    fc_cache, relu_cache = cache
    # fc_cache is the affine_forward cached inputs; cache = (x, w, b) 
    # relu_cache is the relu_forward cached input; cache = a 

    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db


# if __name__ == "__main__":
    # # affine_relu_backward
    # np.random.seed(231)
    # x = np.random.randn(2, 3, 4)
    # w = np.random.randn(12, 10)
    # b = np.random.randn(10)
    # dout = np.random.randn(2, 10)

    # out, cache = affine_relu_forward(x, w, b)
    # dx, dw, db = affine_relu_backward(dout, cache)

   

    