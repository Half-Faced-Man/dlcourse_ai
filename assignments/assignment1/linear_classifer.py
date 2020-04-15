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
    #print(predictions_copy)
    #print(predictions_copy.ndim)
    if predictions_copy.ndim == 1:
        ff = 0
    else:
        ff = 1
        
    probs = np.apply_along_axis( exp_along_row , ff , predictions_copy)
    
    # predictions -= np.max(predictions) 
    # probs = np.exp(predictions) / np.sum(np.exp(predictions)) 
    
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

    # loss = np.apply_along_axis( indexing_along_row , 0 , (probs, target_index) )
    if probs.ndim == 1:
        mask = np.zeros_like(probs)
        mask[target_index] = 1
        mask = mask.astype(bool)
        #print('if')
    else:
        mask = np.zeros_like(probs)
        mask[ np.arange(target_index.shape[0])  ,  target_index.reshape(target_index.shape[0] )] =1 
        mask = mask == True
        #print('else')
    
    #print(probs.shape)
    #print(probs)
    #print(type(mask))
    #print(mask)
    loss = probs[mask]
    #print(loss)
    
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
    # TODO implement softmax with cross-entropy
    # Your final implementation shouldn't have any loops
    
    probs = softmax(predictions)
    #print(probs)
    #print(target_index)
    loss = cross_entropy_loss(probs, target_index)
    
    if probs.ndim == 1:
        target_mask = np.zeros_like(probs)
        target_mask[target_index] = 1
        #target_mask = target_mask.astype(bool)
        #print('if')
    else:
        target_mask = np.zeros_like(probs)
        target_mask[ np.arange(target_index.shape[0])  ,  target_index.reshape(target_index.shape[0] )] =1 
        #mask = mask == True
        #target_mask = target_index #== True
        #print('else')
    
    dprediction = probs - target_mask
    
#     print(probs)
#     print(target_mask)
#     print("loss = " ,  loss ) 
#     print(dprediction)
#     print('======================================')

    return loss , dprediction



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

    # TODO: implement l2 regularization and gradient
    # Your final implementation shouldn't have any loops
    loss = 0.5 * reg_strength * np.sum(W**2)
    grad = reg_strength * W

    return loss, grad
    

def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    '''
    predictions = np.dot(X, W)
    #print(predictions)
    #print(predictions.shape)

    # TODO implement prediction and gradient over W
    # Your final implementation shouldn't have any loops
    loss , dprediction = softmax_with_cross_entropy(predictions , target_index)
    
    dW = np.dot( X.T , dprediction)
    
    #print(loss)
    #print(dprediction)
    #print(dprediction.shape)
    #print(dW)
    
    return loss, dW


class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1):
        '''
        Trains linear classifier
        
        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y)+1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)

            # TODO implement generating batches from indices
            # Compute loss and gradients
            # Apply gradient to weights using learning rate
            # Don't forget to add both cross-entropy loss
            # and regularization!
            
#             print(shuffled_indices.shape)
#             print(sections.shape)
#             print(sections)
#             print(len(batches_indices))
#             print(batches_indices[0])
            
            for batch in batches_indices:
                X_train = X[batch]
                y_train = y[batch]
                
                loss, dW = linear_softmax(X_train, self.W, y_train)
                reg_loss , reg_dW = l2_regularization(self.W, reg)
                
                loss = loss + reg_loss
                dW = dW + reg_dW
                
                self.W = self.W - learning_rate * dW
                
                
            loss_history.append(loss)


            # end
            #print("Epoch %i, loss: %f" % (epoch, loss))

        return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''
        y_pred = np.zeros(X.shape[0], dtype=np.int)
        # print(y_pred.shape)

        # TODO Implement class prediction
        # Your final implementation shouldn't have any loops
        predictions = np.dot(X , self.W)
        predictions = softmax(predictions)
        y_pred = np.argsort(predictions , axis = 1)# [: , 0:1]# [: , 0:self.k]
        tst = np.argmax(predictions , axis = 1)
#         print(predictions[:10])
#         print(y_pred[:10])
#         print(tst[:10])
        
        
        return tst

                
                                                          

            

                
