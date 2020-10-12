import numpy as np


def exp_along_row(row):
    row -= np.max(row) 
    row = np.exp(row) / np.sum(np.exp(row))
    return row


def indexing_along_row(data):
    probs , target_index = data
    return probs[target_index]


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


def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    # TODO: Copy from the previous assignment
    loss = 0.5 * reg_strength * np.sum(W**2)
    grad = reg_strength * W
    
    return loss, grad


def softmax_with_cross_entropy(preds, target_index):
    """
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
    """
    # TODO: Copy from the previous assignment
    probs = softmax(preds)
    loss = cross_entropy_loss(probs, target_index)
    
    if probs.ndim == 1:
        target_mask = np.zeros_like(probs)
        target_mask[target_index] = 1
    else:
        target_mask = np.zeros_like(probs)
        target_mask[ np.arange(target_index.shape[0])  ,  target_index.reshape(target_index.shape[0] )] =1 
    
    dprediction = probs - target_mask

    return loss, dprediction # d_preds


# def linear_softmax(X, W, target_index):
#     '''
#     Performs linear classification and returns loss and gradient over W

#     Arguments:
#       X, np array, shape (num_batch, num_features) - batch of images
#       W, np array, shape (num_features, classes) - weights
#       target_index, np array, shape (num_batch) - index of target classes

#     Returns:
#       loss, single value - cross-entropy loss
#       gradient, np.array same shape as W - gradient of weight by loss

#     '''
#     predictions = np.dot(X, W)
#     # TODO implement prediction and gradient over W
#     # Your final implementation shouldn't have any loops
#     loss , dprediction = softmax_with_cross_entropy(predictions , target_index)
    
#     dW = np.dot( X.T , dprediction)
    
#     return loss, dW


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
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
        # TODO: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment

        # loss, dW = linear_softmax(X_train, self.W, y_train)
        # reg_loss , reg_dW = l2_regularization(self.W, reg)
        # loss = loss + reg_loss
        # dW = dW + reg_dW
        # self.W = self.W - learning_rate * dW
        
        
        self.B.grad = np.sum( np.multiply(d_out , np.ones_like(d_out)) , axis = 0 , keepdims = True)  # dB = d_out * I
        self.W.grad = np.dot(self.X.value.T , d_out) # dW = Xt * d_out

        self.X.grad = np.dot(d_out , self.W.value.T) # dX = d_out * Wt
        
        d_result = self.X.grad
        
        return d_result

    def params(self):
        return {'W': self.W, 'B': self.B}
