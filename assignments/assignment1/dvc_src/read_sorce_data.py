import numpy as np
import sys
import os
# sys.path.append('../')
sys.path.append('/Users/a17409438/Documents/dlcourse_ai/assignments/assignment1/')
from dataset import load_svhn


def load_source_data():
    
    # path, max_train=1000, max_test=100
    # train_X, train_y, test_X, test_y = load_svhn(path, max_train=1000, max_test=100)
    train_X, train_y, test_X, test_y = load_svhn("data", max_train=1000, max_test=100)

    
    return train_X, train_y, test_X, test_y


if __name__ == '__main__':
    
    train_X, train_y, test_X, test_y = load_source_data()
    
    train_X = train_X.reshape(train_X.shape[0], -1)
    test_X = test_X.reshape(test_X.shape[0], -1)

    np.savetxt('data/train_X.csv' , train_X , delimiter=",")
    np.savetxt('data/train_y.csv' , train_y , delimiter=",")
    np.savetxt('data/test_X.csv' , test_X , delimiter=",")
    np.savetxt('data/test_y.csv' , test_y , delimiter=",")