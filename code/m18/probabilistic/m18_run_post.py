"""
This is the script which runs through posterior distributions, trains the model on each, and tests. All other hyperparameters are found in hpPost.py
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
import hpPost #Import hyperparamters

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
datasetVal = dataRawWav.datasetTrain.skip(hpPost.trainNo)
datasetTrain = dataRawWav.datasetTrain.take(hpPost.trainNo)
datasetTest = dataRawWav.datasetTest
#Create batches
batchTrain = datasetTrain.batch(hpPost.batchSize, drop_remainder=False) #Suspect that drop_remainder=True works better
batchVal = datasetVal.batch(hpPost.batchSize, drop_remainder=False)
batchTest = datasetTest.batch(hpPost.batchSize, drop_remainder=False)

posteriorModule = __import__('posteriorDists', globals(), locals(), [], 0)
allVarsLst = list(posteriorModule.__dict__.keys())
posteriors = [x for x in allVarsLst if 'fn' == x[-2:]]

for posterior in posteriors:
    print("Starting training for posterior: " + str(posterior)+"------------------------------------------")
    #Define the posterior function
    post = posteriorModule.__dict__[posterior]
    post = post()

    #This function ceates a model based on the inputted posterior function and returns it.
    import loadModels.load18pLayerPosteriorFunction
    m = loadModels.load18pLayerPosteriorFunction.returnModel(post)
    
    #Set model and fit
    history = m.fit(batchTrain,
        epochs=hpPost.epochs,
        validation_data = batchVal)
    
    print("Finished training for posterior: " + str(posterior)+"------------------------------------------")
    
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
            hpPost.klMcSampleNo, 
            hpPost.epochs,
            hpPost.learningRate, 
            hpPost.batchSize,
            hpPost.SCALE1, 
            hpPost.SCALE2, 
            hpPost.PI,
            hpPost.posterior,
            hpPost.prior,
            hpPost.divFn)

    resultsWriter.resultsWriterCsvPosterior(posterior, testAccuracies, trainAccuracyHist, valAccuracyHist, hpPost.prior)

    print("Finished writing results for posterior: " + str(posterior)+"------------------------------------------")
