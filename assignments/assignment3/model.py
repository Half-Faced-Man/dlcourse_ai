import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener, softmax,
    softmax_with_cross_entropy, l2_regularization
    )

from gradient_check import check_layer_gradient, check_layer_param_gradient, check_model_gradient


            
class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """
    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """
        self.input_shape = input_shape
        self.n_output_classes = n_output_classes
        self.conv1_channels = conv1_channels
        self.conv2_channels = conv2_channels
        # TODO Create necessary layers
        # raise Exception("Not implemented!")
        
        image_width , image_height , n_channels = self.input_shape
        mp1 = 4
        mp2 = 4
        
        self.conv1 = ConvolutionalLayer(in_channels=n_channels, out_channels=self.conv1_channels,
                                        filter_size=3, padding=1)
        self.relu1 = ReLULayer()
        self.maxpool1 = MaxPoolingLayer(mp1, mp1)
        
        self.conv2 = ConvolutionalLayer(in_channels=self.conv1_channels, out_channels=self.conv2_channels,
                                        filter_size=3, padding=1)
        self.relu2 = ReLULayer()
        self.maxpool2 = MaxPoolingLayer(mp2, mp2)
        
        self.flat = Flattener()
        
        ## тут надо поменять размеры слоя учитывая все измениня в CL и в MPL
        fc_width = image_width / (mp1 * mp2 )
        fc_height = image_height / (mp1 * mp2 )
        fc_size = int( fc_width * fc_height * self.conv2_channels ) 
        
        
        self.fc = FullyConnectedLayer(fc_size , self.n_output_classes)
        
        self.layers_list = [ self.conv1 ,
                             self.relu1 ,
                             self.maxpool1 ,
                             self.conv2 ,
                             self.relu2 ,
                             self.maxpool2 ,
                             self.flat ,
                             self.fc ]
      
        

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass

        # TODO Compute loss and fill param gradients
        # Don't worry about implementing L2 regularization, we will not
        # need it in this assignment
        # raise Exception("Not implemented!")
        
        
        for k , v in self.params().items():
            self.params()[k].grad = np.zeros_like(self.params()[k].grad)
        

        X_local = X.copy()
        y_local = y.copy()
        for i , layer in enumerate( self.layers_list ):
            # print(i)
            # assert check_layer_gradient(layer, X_local)
            X_local = layer.forward(X_local)
            # print(X_local.shape)

        loss , dprediction = softmax_with_cross_entropy(X_local , y_local)
        dCE = np.dot( X.T , dprediction)

        # print('==========================================================')

        for layer in self.layers_list[::-1]:
            dprediction = layer.backward(dprediction)

         #print('==========================================================')

#         for layer in self.layers_list[::-1]:
#             for i in layer.params().keys():
#                 reg_loss , reg_dprediction = l2_regularization(layer.params()[i].value , self.reg)
#                 layer.params()[i].grad = layer.params()[i].grad + reg_dprediction
#                 loss = loss + reg_loss
        

        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        # raise Exception("Not implemented!")

        return loss
        

    def predict(self, X):
        # You can probably copy the code from previous assignment
        # raise Exception("Not implemented!")
        
        pred = np.zeros(X.shape[0], np.int)

        X_pred = X.copy()
        for layer in self.layers_list:
            X_pred = layer.forward(X_pred)

        pred = softmax(X_pred)
        pred = np.argmax(pred , axis = 1)

        return pred
    

    def params(self):
        result = {}

        # TODO: Aggregate all the params from all the layers
        # which have parameters
        # raise Exception("Not implemented!")

        for i , layer in enumerate( self.layers_list ):
            for k , v in layer.params().items():
                result['{}_{}'.format(i , k)] = v

        return result
