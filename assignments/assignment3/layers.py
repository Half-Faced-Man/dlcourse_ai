import numpy as np


def exp_along_row(row):
    row -= np.max(row) 
    row = np.exp(row) / np.sum(np.exp(row))
    return row


def indexing_along_row(data):
    probs , target_index = data
    return probs[target_index]


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''
    # TODO: Copy from previous assignment
    # raise Exception("Not implemented!")

    loss = 0.5 * reg_strength * np.sum(W**2)
    grad = reg_strength * W
    
    return loss, grad


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    # TODO implement softmax
    # Your final implementation shouldn't have any loops
    
    predictions_copy = predictions.copy()
    if predictions_copy.ndim == 1:
        ff = 0
    else:
        ff = 1
        
    probs = np.apply_along_axis( exp_along_row , ff , predictions_copy)
    
    return probs


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    # TODO implement cross-entropy
    # Your final implementation shouldn't have any loops

    if probs.ndim == 1:
        mask = np.zeros_like(probs)
        mask[target_index] = 1
        mask = mask.astype(bool)
    else:
        mask = np.zeros_like(probs)
        mask[ np.arange(target_index.shape[0])  ,  target_index.reshape(target_index.shape[0] )] =1 
        mask = mask == True
    
    loss = probs[mask]
    
    return (-1) * np.sum( np.log(loss) ) 


def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    # TODO copy from the previous assignment
    probs = softmax(predictions)
    loss = cross_entropy_loss(probs, target_index)
    
    if probs.ndim == 1:
        target_mask = np.zeros_like(probs)
        target_mask[target_index] = 1
    else:
        target_mask = np.zeros_like(probs)
        target_mask[ np.arange(target_index.shape[0])  ,  target_index.reshape(target_index.shape[0] )] =1 
    
    dprediction = probs - target_mask

    return loss, dprediction # d_preds


class Param:
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''
    def __init__(self, value):
        self.value = value.copy()
        self.grad = np.zeros_like(value)

        
class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        # TODO: Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        local_X = X.copy()
        self.d_r = (local_X > 0).astype(int)
        return self.d_r * local_X
        
    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Your final implementation shouldn't have any loops
        
        return np.multiply(d_out , self.d_r)

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO: Implement forward pass
        # Your final implementation shouldn't have any loops
        self.X = Param(X)
        return np.add( np.dot(X , self.W.value) , self.B.value )

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        
        # self.B.grad = np.sum( np.multiply(d_out , np.ones_like(d_out)) , axis = 0 , keepdims = True)  # dB = d_out * I
        # self.W.grad = np.dot(self.X.value.T , d_out) # dW = Xt * d_out
        #
        # self.X.grad = np.dot(d_out , self.W.value.T) # dX = d_out * Wt

        self.B.grad = np.sum( np.multiply(d_out , np.ones_like(d_out)) , axis = 0 , keepdims = True)  # dB = d_out * I
        self.W.grad = np.dot(self.X.value.T , d_out) # dW = Xt * d_out

        self.X.grad = np.dot(d_out , self.W.value.T) # dX = d_out * Wt
        
        d_result = self.X.grad


################# Добавил +=

#         self.B.grad += np.sum( np.multiply(d_out , np.ones_like(d_out)) , axis = 0 , keepdims = True)  # dB = d_out * I
#         self.W.grad += np.dot(self.X.value.T , d_out) # dW = Xt * d_out

#         self.X.grad = np.dot(d_out , self.W.value.T) # dX = d_out * Wt
        
#         d_result = self.X.grad
        
        return d_result

    def params(self):
        return {'W': self.W, 'B': self.B}



    

class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))

        self.padding = padding


    def forward(self, X):
        batch_size, height, width, channels = X.shape
        
        # TODO: Implement forward pass
        # Hint: setup variables that hold the result
        # and one x/y location at a time in the loop below
        
        # It's ok to use loops for going over width and height
        # but try to avoid having any other loops
#         for y in range(out_height):
#             for x in range(out_width):
#                 # TODO: Implement forward pass for specific location
#                 pass
        # raise Exception("Not implemented!")
    
        out_height = ( height - self.filter_size + 2 * self.padding ) / 1 + 1
        out_height = int(out_height)
        out_width = ( width - self.filter_size + 2 * self.padding ) / 1 + 1
        out_width = int(out_width)
        
        new_X = np.zeros(shape = (batch_size , height+ 2 * self.padding ,
                                  width + 2 * self.padding , channels))

        new_X[: , self.padding: height +self.padding , self.padding: width +self.padding , :] = X
        
        self.X = Param(X)
        self.new_X = Param(new_X)
        
        res = np.zeros((batch_size ,  out_height , out_width , self.out_channels))
        
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement forward pass for specific location
                env = new_X[: , y: y + self.filter_size, x: x + self.filter_size , :].\
                        reshape( (batch_size, self.filter_size*self.filter_size*self.in_channels) )

                w = self.W.value.reshape( (self.filter_size*self.filter_size*self.in_channels, self.out_channels) )
                
                # tmp = np.dot(env , w) + self.B.value
                tmp = np.add( np.dot(env , w) , self.B.value )
                
                res[: , y , x , :] = tmp

        return res
        
    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients

        batch_size, height, width, channels = self.X.value.shape
        _, out_height, out_width, out_channels = d_out.shape

        # TODO: Implement backward pass
        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output

        # Try to avoid having any other loops here too
#         for y in range(out_height):
#             for x in range(out_width):
#                 # TODO: Implement backward pass for specific location
#                 # Aggregate gradients for both the input and
#                 # the parameters (W and B)
#                 pass


        for y in range(out_height):
            for x in range(out_width):
                d_out_loop = d_out[: , y , x , :]
                d_out_loop = d_out_loop.reshape((-1 , self.out_channels ))
                
                W_loop = self.W.value.reshape((self.filter_size * self.filter_size * self.in_channels , self.out_channels))
                
                d_X = np.dot(d_out_loop , W_loop.T)
                d_X = d_X.reshape(batch_size, self.filter_size , self.filter_size , self.in_channels)
                self.new_X.grad[: , y:y + self.filter_size , x:x + self.filter_size , :] += d_X  #  += d_X
                
                X_loop = self.new_X.value[: , y:y + self.filter_size , x:x + self.filter_size , :].reshape((batch_size , self.filter_size*self.filter_size*self.in_channels))
                d_W = np.dot(X_loop.T , d_out_loop )
                self.W.grad += d_W.reshape((self.filter_size, self.filter_size, self.in_channels, self.out_channels))
                
                self.B.grad += np.sum(d_out_loop , axis = 0 )
            
        self.X.grad = self.new_X.grad[: , self.padding: height +self.padding , self.padding: width +self.padding , :]
        d_result = self.X.grad
        # self.B.grad
        
        return d_result
                


    def params(self):
        return { 'W': self.W, 'B': self.B }
    
    
    

class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        # TODO: Implement maxpool forward pass
        # Hint: Similarly to Conv layer, loop on
        # output x/y dimension
        # raise Exception("Not implemented!")
        
        self.X = X.copy()
        
        out_height = int( ( height - self.pool_size + self.stride ) / self.stride ) 
        out_width = int( ( width - self.pool_size + self.stride ) / self.stride )
        
        out = np.zeros(shape = (batch_size ,out_height , out_width , channels ) )
        
        for h in range(out_height):
            for w in range(out_width):
#                 print( self.X[: , h:h+self.pool_size , w:w+self.pool_size , :] )
#                 print( np.max( self.X[: , h:h+self.pool_size , w:w+self.pool_size , :] ,axis = (1,2) ) )
#                 print('---')
                
#                 print(list(range(out_height)))
#                 print(list(range(out_width)))
#                 print(h , w)
                
                out[: , h , w , :] = np.max( self.X[: , h*self.stride : h*self.stride+self.pool_size , w*self.stride : w*self.stride+self.pool_size , :] , axis = (1,2) )
                
        return out 

    def backward(self, d_out):
        # TODO: Implement maxpool backward pass
        batch_size, height, width, channels = self.X.shape
        # raise Exception("Not implemented!")
        
        # _ , out_height, out_width, _ = d_out.shape
        out_height = int( ( height - self.pool_size + self.stride ) / self.stride ) 
        out_width = int( ( width - self.pool_size + self.stride ) / self.stride )
        
        result = np.zeros(shape = (batch_size, height, width, channels) )
        
        for b in range(batch_size):
            for c in range(channels):

                for h in range(out_height):
                    for w in range(out_width):
                        # pool = self.X[b , h:h+self.pool_size , w:w+self.pool_size , c]
                        pool = self.X[b , h*self.stride : h*self.stride+self.pool_size , w*self.stride : w*self.stride+self.pool_size , c]

                        max_ind_2d = np.unravel_index(np.argmax(pool, axis=None), pool.shape)

#                         print(max_ind_2d)
                        
#                         print(result[b , h:h+self.pool_size , w:w+self.pool_size , c])
                        
#                         print(result[b , h:h+self.pool_size , w:w+self.pool_size , c][max_ind_2d[0] , max_ind_2d[1]])
                        
                        # result[b , h:h+self.pool_size , w:w+self.pool_size , c][max_ind_2d[0] , max_ind_2d[1]] = d_out[b , h , w , c]
                        result[b , h*self.stride : h*self.stride+self.pool_size ,
                                w*self.stride : w*self.stride+self.pool_size , c][max_ind_2d[0], max_ind_2d[1]] =\
                                d_out[b, h, w, c]  # =\

                        # out[: , h , w , :] = np.max( self.X[: , h:h+self.pool_size , w:w+self.pool_size , :] , axis = (1,2) )

        return result
        

    def params(self):
        return {}

    

class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape

        # TODO: Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]
        # raise Exception("Not implemented!")
        self.X_shape = X.shape
        
        return X.reshape(batch_size, height * width * channels)

    def backward(self, d_out):
        # TODO: Implement backward pass
        # raise Exception("Not implemented!")
        
        return d_out.reshape(self.X_shape)

    def params(self):
        # No params!
        return {}
