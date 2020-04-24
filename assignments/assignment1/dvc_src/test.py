import numpy as np
import pickle
import json
import sys
import os
# sys.path.append('../')
sys.path.append('/Users/a17409438/Documents/dlcourse_ai/assignments/assignment1/')
from knn import KNN
from metrics import multiclass_accuracy


if __name__ == '__main__':
    
    with open('model/dvc_knn_model' , 'rb') as m:
        model = pickle.load(m)
    
    test_X = np.genfromtxt('data/test_X.csv',delimiter=",")
    test_y = np.genfromtxt('data/test_y.csv',delimiter=",")

    predict = model.predict(test_X)
    
    accuracy = multiclass_accuracy(predict, test_y)
    print("Accuracy: %4.2f" % accuracy)
    
    json.dump(
        obj={
            'Accuracy': accuracy
        },
        fp=open('data/accuracy.txt', 'w')
    )