import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization
from layers import softmax


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        self.n_input = n_input
        self.n_output = n_output
        self.hidden_layer_size = hidden_layer_size
        # TODO Create necessary layers
        self.hidden_layer = FullyConnectedLayer(self.n_input, self.hidden_layer_size)
        self.relu_layer = ReLULayer()
        self.output_layer = FullyConnectedLayer(self.hidden_layer_size, self.n_output)

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        # Hint: using self.params() might be useful!

        for k , v in self.params().items():
            self.params()[k].grad = np.zeros_like(self.params()[k].grad)
            # print(self.params()[k].grad)
            # print(self.params()[k].grad.shape)
        
        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model

        X_local = X.copy()
        y_local = y.copy()
        for layer in [self.hidden_layer,
                      self.relu_layer,
                      self.output_layer]:
            # print(X_local.shape)
            X_local = layer.forward(X_local)

        # print(X_local.shape)

        # print(X_local)
        # print(y_local)

        # print('==========================================================')
        loss , dprediction = softmax_with_cross_entropy(X_local , y_local)
        # print(dprediction.shape)
        dCE = np.dot( X.T , dprediction)
        # print(dCE.shape)


        # print('==========================================================')

        for layer in [self.hidden_layer,
                      self.relu_layer,
                      self.output_layer][::-1]:
            # print(dprediction.shape)
            dprediction = layer.backward(dprediction)
        
        # print(dprediction.shape)

        # print('==========================================================')

        for layer in [self.hidden_layer,
                      self.relu_layer,
                      self.output_layer][::-1]:
            # print(layer.params()['W'])
            # print(layer.params()['B'])
            for i in layer.params().keys():
                # print(layer.params()[i].value)
                reg_loss , reg_dprediction = l2_regularization(layer.params()[i].value , self.reg)
                layer.params()[i].grad = layer.params()[i].grad + reg_dprediction
                loss = loss + reg_loss
                # print(reg_dprediction.shape)
                # print(reg_dprediction)
        

        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        # raise Exception("Not implemented!")

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        pred = np.zeros(X.shape[0], np.int)

        X_pred = X.copy()
        for layer in [self.hidden_layer,
                      self.relu_layer,
                      self.output_layer]:
            X_pred = layer.forward(X_pred)

        pred = softmax(X_pred)
        pred = np.argmax(pred , axis = 1)

        return pred

    def params(self):
        result = {}
        # TODO Implement aggregating all of the params
        # result = {'hidden_layer_W_value': self.hidden_layer.W.value,
        #           'hidden_layer_W_grad': self.hidden_layer.W.grad,
        #           'hidden_layer_B_value': self.hidden_layer.B.value,
        #           'hidden_layer_B_grad': self.hidden_layer.B.grad,
        #           'output_layer_W_value': self.output_layer.W.value,
        #           'output_layer_W_grad': self.output_layer.W.grad,
        #           'output_layer_B_value': self.output_layer.B.value,
        #           'output_layer_B_grad': self.output_layer.B.grad
        #           }
        for i , layer in enumerate( [self.hidden_layer,
                  self.relu_layer,
                  self.output_layer] ):
            # print(i , layer)
            for k , v in layer.params().items():
                result['{}_{}'.format(i , k)] = v
                # for o , p in v.__dict__.items():
                #     result['{}_{}_{}'.format(i , k , o)] = p

        # raise Exception("Not implemented!")

        return result
