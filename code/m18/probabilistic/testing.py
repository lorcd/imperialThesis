#Testing the model, seeing what the outputs look like.
import tensorflow as tf
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import numpy as np
import statistics

"""
This script is used to test the accuracy on the test set of a model.
"""
#First test fn is overwritten by second
def test(x, y, m, n):
    """
    outputs two accuracies: one calculated based on the predictions garnered from
    averaging n softmax distributions of event size 10, and one garnered from taking the mode
    of the labels predicted based on the n softmax distributions.
    """
    preds = [] 
    predLabels = [] 
    accs=[]
    
    for i in range(n):
        pred = m(x, training=False)
        preds.append(pred.numpy())
        predLabel = np.argmax(pred.numpy(), axis=1)
        predLabels.append(predLabel)
    
    finalPred = np.mean(np.array(preds), axis=0)
    finalPredLabel1 = np.argmax(finalPred, axis = 1)
    finalPredLabel2 = np.array([statistics.mode(i) for i in np.array(predLabels).T])
    #Calculate accuracy against y
    totalNumCorrect1 = sum(finalPredLabel1 == y)
    totalNumCorrect2 = sum(finalPredLabel2 == y)
    totalNum = x.shape[0]
    acc1, acc2 = totalNumCorrect1/totalNum, totalNumCorrect2/totalNum
    return acc1

def test(x, y, m, n):
    """
    outputs accuracy: calculated averaging n softmax distributions each of event size 10.
    """
    preds = []
    predLabels = []
    accs=[]
    
    for i in range(n):
        pred = m(x, training=False)
        preds.append(pred.numpy())
        predLabel = np.argmax(pred.numpy(), axis=1)
        predLabels.append(predLabel)
    
    finalPred = np.mean(np.array(preds), axis=0)
    finalPredLabel = np.argmax(finalPred, axis = 1)
    #Calculate accuracy against y
    totalNumCorrect = sum(finalPredLabel == y)
    totalNum = x.shape[0]
    acc = totalNumCorrect/totalNum
    return acc

def testMetrics(dataSet, m, numsPredictionsToAverage = [1, 2, 5, 10]):
    x, y = list(dataSet.batch(1000))[0] #We just take 1000 cuz it's bigger than test set size
    x = x.numpy()
    y = y.numpy()
    accList = []
    for i in numsPredictionsToAverage:
        acc = test(x, y, m, i)
        accList.append(acc)
    return accList 
    
def oneHotEncoderElem(i, n=10):
    a = np.repeat(0, n)
    a[i] = 1
    return list(a)

def oneHotEncoder(y):
    return np.array([oneHotEncoderElem(i) for i in y])
