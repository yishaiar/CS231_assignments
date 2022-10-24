from builtins import range
from builtins import object
import numpy as np

# from layers import *
# from layer_utils import *
from cs231n.layers import *
from cs231n.layer_utils import *





class FullyConnectedNet(object):
    """Class for a multi-layer fully connected neural network.

    Network contains an arbitrary number of hidden layers, ReLU nonlinearities,
    and a softmax loss function. This will also implement dropout and batch/layer
    normalization as options. For a network with L layers, the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional and the {...} block is
    repeated L - 1 times.

    Learnable parameters are stored in the self.params dictionary and will be learned
    using the Solver class.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout_keep_ratio=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer[dim1,dim2,..dim_last]
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout_keep_ratio: Scalar between 0 and 1 giving dropout strength.
            If dropout_keep_ratio=1 then the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
            are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
            initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
            this datatype. float32 is faster but less accurate, so you should use
            float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers.
            This will make the dropout layers deteriminstic so we can gradient check the model.
            using np.random.seed(seed) in both places would result with same matrix (layer)
        """
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1 #true/false
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
           
        print('normalization = ',self.normalization)
        print('drop out: ',self.use_dropout)
       
        #  affine forward
       
        # input dim of first layer is; input_dim = np.prod(input_shape)
        #  append first layer dim to hidden dim 
        hidden_dims.insert(0, input_dim)
        # append last layer dim to hidden dim; layer output is number of classes
        hidden_dims.append(num_classes)
        #  the input dim of second layer is first  hidden_dim
        # the output dim of second layer is second  hidden_dim

        #ind is layer index; initilized at layer 1 
        for ind,dim_0,dim_1 in zip (range(1,self.num_layers+1),hidden_dims[:-1],hidden_dims[1:]):
            
            self.params['W' + str(ind)] = np.random.normal(0, weight_scale, (dim_0, dim_1))
            # * sqrt(2.0/n)
            self.params['b'+ str(ind)] = np.zeros((1,dim_1))
            
            if self.normalization != None: 
            #there are no beta gamma in model (in that case they were never used )
       
                # beta is E(x) i.e the shift parmater
                self.params['beta'+ str(ind)] = np.zeros((1,dim_1))
                # gamma is sqrt(var(x)) i.e the Scale parameters
                self.params['gamma'+ str(ind)] = np.ones((1,dim_1))
        # del self.params['b'+ str(ind)] # last layer without bias

        


        
        
        
        
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed


        
        
        # normalization types: "batchnorm", "layernorm", or None for no normalization (the default).
        # 1 -  batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # 2 - normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]
        

        # Cast all parameters to the correct datatype.
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

            

    def loss(self, X, y=None):
        """Compute loss and gradient for the fully connected net.
        
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
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        # LAYERS:  {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax
        # where batch/layer normalization and dropout are optional and the {...} block is
        # repeated L - 1 times.
        # If y is None (we are in test mode)  just return scores (no need for softmax)
        
        # # state
        # print ('mode is: ', mode)
        
        
        # save cach of layers for back propogation
        cache = {}

        # the output scores of previous layer is entered to next layer, the new layer output is run over the previous
        # first layer with out = X 
        scores = X.copy()


        # we need to pass a special bn_param object to each batch when normalizing (written previously)
        # when normalizing ==NONE -> bn_params == NONE:
        if self.normalization == None:
            self.bn_params = [None for i in range(self.num_layers - 1)]
        
        #ind is layer index; initilized at layer 1 run untill layer (L-1)
        for ind in range(1,self.num_layers):
            
            # affine_relu_forward is affine - relu (missing normalization) 
            # scores, cache['cache' + str(ind)] = affine_relu_forward(scores,self.params['W' + str(ind)] , self.params['b' + str(ind)])
            # affine_norm_activation_forward is affine - norm - activation(relu) 
            if self.normalization != None: 
                scores, cache['cache' + str(ind)] = affine_norm_activation_forward(scores,self.params['W' + str(ind)] \
                    ,self.params['b' + str(ind)],self.bn_params[ind-1], self.normalization, 'ReLU'\
                    ,self.params['beta' + str(ind)],self.params['gamma' + str(ind)])
            else:
                scores, cache['cache' + str(ind)] = affine_norm_activation_forward(scores,self.params['W' + str(ind)] \
                    ,self.params['b' + str(ind)],self.bn_params[ind-1], self.normalization, 'ReLU')

            # print('layer '+ str(ind), ': forward affine - norm - relu ')

            # drop out
            if self.use_dropout:
                scores, cache['dropout_cache' + str(ind)] = dropout_forward(scores, self.dropout_param)
                # print('layer '+ str(ind), ': forward dropout ')

            
            # if self.normalization == 'batchnorm':
            #     pass
            # elif self.normalization == 'layernorm':
            #     pass
            # else: #if self.normalization == None:
            #     pass
                

        # affine_forward is  affine only (without relu, drop out)
        ind += 1
        scores, cache['cache' + str(ind)] = affine_forward(scores,self.params['W' + str(ind)] , self.params['b' + str(ind)])
        # print('layer '+ str(ind), ': forward affine only')
        
        # output scores
       
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early.
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the   #
        # scale and shift parameters.                                              #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # calculate loss  - according to the forward pass scores
        loss,dloss = softmax_loss(scores, y)
        for ind in range (1,self.num_layers+1):
            # print('loss regularization by layer: ',ind)
            W =  self.params['W' + str(ind)]
            loss += 0.5 * self.reg * np.sum(W * W)
        
        
        
        # # backpropogation of a multi layer neural network: 
        
        # # last layer - recieve the gradiant with respect of x from the final output (dloss)
        ind = self.num_layers #start from last layer
        # print('layer ' + str(ind), ': backward affine only')
        d_scores, dW, db = affine_backward(dloss, cache['cache' + str(ind)])
        # save gradiants
        grads['W'+ str(ind)] = dW + self.reg * self.params['W' + str(ind)]
        grads['b'+ str(ind)] = db.astype(self.dtype)
        
        # rest of layeres- each recieve the gradiant with respect of its previous layer (d_out)
        for ind in range(self.num_layers-1,0,-1):
            # drop out
            if self.use_dropout:
                # print('layer '+ str(ind), ': backward dropout ')
                d_scores = dropout_backward(d_scores, cache['dropout_cache' + str(ind)])
                

            # print('layer ' + str(ind), ': backward relu - affine')
            # d_scores, dW, db = affine_relu_backward(d_scores, cache['cache' + str(ind)]) 
            d_scores, dW, db = affine_norm_activation_backward(d_scores, cache['cache' + str(ind)],norm = self.normalization, activation = 'ReLU')
            
            # save gradiants
            W =  self.params['W' + str(ind)]
            grads['W'+ str(ind)] = dW + self.reg * W
            grads['b'+ str(ind)] = db
        # self.grads = grads
        
        del cache
             
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


    




if __name__ == "__main__":
    
    

    np.random.seed(231)
    N, D, H1, H2, C = 2, 15, 20, 30, 10
    X = np.random.randn(N, D)
    y = np.random.randint(C, size=(N,))

    for reg in [0, 3.14]:
        print("Running check with reg = ", reg)
        model = FullyConnectedNet(
            [H1, H2],
            input_dim=D,
            num_classes=C,
            reg=reg,
            weight_scale=5e-2,
            dtype=np.float64,

            dropout_keep_ratio=0.9,
            # normalization=None,
            normalization='batchnorm',
            # normalization='layernorm',

        )


        loss, grads = model.loss(X, y)
        # scores = model.loss(X)
        print("Initial loss: ", loss)

        # Most of the errors should be on the order of e-7 or smaller.   
        # NOTE: It is fine however to see an error for W2 on the order of e-5
        # for the check when reg = 0.0
        p = sorted(grads)
        print(p)
        
        for name in sorted(grads):
            print(name)
            f = lambda _: model.loss(X, y)[0]
            grad_num = eval_numerical_gradient(f, model.params[name], verbose=False, h=1e-5)
            print(f"{name} relative error: {rel_error(grad_num, grads[name])}")