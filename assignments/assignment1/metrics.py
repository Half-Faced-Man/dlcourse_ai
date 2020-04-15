import numpy as np


def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    
#     res_plus = prediction.astype(int) + ground_truth.astype(int)
#     res_minus  = prediction.astype(int) - ground_truth.astype(int)
#     res = np.array([res_plus , res_minus]).T
#     val , count = np.unique(res , return_counts=True , axis = 0)
    
#     tn_mask = np.equal(val , [0,0]).all(axis = 1) 
#     fp_mask = np.equal(val , [1,1]).all(axis = 1) 
#     fn_mask = np.equal(val , [1,-1]).all(axis = 1) 
#     tp_mask = np.equal(val , [2,0]).all(axis = 1) 
    
#     tn = count[tn_mask] 
#     fp = count[fp_mask]
#     fn = count[fn_mask]
#     tp = count[tp_mask]

    tp = float(np.sum((prediction == 1) & (ground_truth == 1)))
    tn = float(np.sum((prediction == 0) & (ground_truth == 0)))
    fp = float(np.sum((prediction == 1) & (ground_truth == 0)))
    fn = float(np.sum((prediction == 0) & (ground_truth == 1)))


    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    # TODO: Implement computing accuracy
    
    accuracy = float(np.sum(prediction == ground_truth )) / ground_truth.size
    
    return accuracy
