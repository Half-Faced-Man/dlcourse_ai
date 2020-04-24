import numpy as np
import pickle
import sys
import os
# sys.path.append('../')
sys.path.append('/Users/a17409438/Documents/dlcourse_ai/assignments/assignment1/')
from knn import KNN


if __name__ == '__main__':
    
    train_X = np.genfromtxt('data/train_X.csv',delimiter=",")
    train_y = np.genfromtxt('data/train_y.csv',delimiter=",")

    knn_classifier = KNN(k=5)
    knn_classifier.fit(train_X, train_y)
    
    with open('model/dvc_knn_model' , 'wb') as m:
        pickle.dump(knn_classifier, m)