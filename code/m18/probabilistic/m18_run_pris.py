"""
This is the script which runs through prior distributions, trains the model on each, and tests. All other hyperparameters are found in hpPri.py
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models

from keras.layers.core import Activation, Dense
from keras.layers.normalization import BatchNormalization
from keras import regularizers

import cv2
import numpy as np
import sys #Use this to automate which model gets run
import hpPri #Import hyperparamters

numProbLayers = sys.argv[1] #string
audioLength = 32000
numberClasses = 10

#Testing the model, seeing what the outputs look like.
import testing

#We will write these results to a file named after the model
modelName = "m18model_"+numProbLayers+"pLayers"
import resultsWriter

#Load in the data.
import dataRawWav 
#We'll also split the training set for validation
datasetVal = dataRawWav.datasetTrain.skip(hpPri.trainNo)
datasetTrain = dataRawWav.datasetTrain.take(hpPri.trainNo)
datasetTest = dataRawWav.datasetTest
#Create batches
batchTrain = datasetTrain.batch(hpPri.batchSize, drop_remainder=False) #Suspect that drop_remainder=True works better
batchVal = datasetVal.batch(hpPri.batchSize, drop_remainder=False)
batchTest = datasetTest.batch(hpPri.batchSize, drop_remainder=False)

priorModule = __import__('priorDists', globals(), locals(), [], 0)
allVarsLst = list(priorModule.__dict__.keys())
priors = [x for x in allVarsLst if 'fn' == x[-2:]]

for prior in priors:
    print("Starting training for prior: " + str(prior)+"------------------------------------------")
    #Define the prior function
    pri = priorModule.__dict__[prior]

    #This function ceates a model based on the inputted prior function and returns it.
    import loadModels.load18pLayerPriorFunction
    m = loadModels.load18pLayerPriorFunction.returnModel(pri)
    
    #Set model and fit
    history = m.fit(batchTrain,
        epochs=hpPri.epochs,
        validation_data = batchVal)
    
    print("Finished training for prior: " + str(prior)+"------------------------------------------")
    
    #Testing
    noSamples = (10, 10, 10, 10, 10) #All ten MC samples
    valAccuracies = testing.testMetrics(datasetVal, m, noSamples)
    testAccuracies = testing.testMetrics(datasetTest, m, noSamples)
    trainAccuracyFinal = history.history['accuracy'][-1]
    trainAccuracyHist = history.history['accuracy']
    valAccuracyHist = history.history['val_accuracy']
    print(testAccuracies)
    
    resultsWriter.resultsWriter(valAccuracies, testAccuracies, 'Nan', 'Nan',
            modelName, 
            hpPri.klMcSampleNo, 
            hpPri.epochs,
            hpPri.learningRate, 
            hpPri.batchSize,
            hpPri.SCALE1, 
            hpPri.SCALE2, 
            hpPri.PI,
            hpPri.posterior,
            hpPri.prior,
            hpPri.divFn)

    resultsWriter.resultsWriterCsvPrior(prior, testAccuracies, trainAccuracyHist, valAccuracyHist)

    print("Finished writing results for prior: " + str(prior)+"------------------------------------------")
