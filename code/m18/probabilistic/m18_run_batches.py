"""
This is the script which runs through batch values, trains the model on each, and tests. All other hyperparameters are found in hpBatch.py
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models

from keras.layers.core import Activation, Dense
from keras.layers.normalization import BatchNormalization
from keras import regularizers

from keras.callbacks import Callback
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

import cv2
import numpy as np
import sys #Use this to automate which model gets run
import hpBatch #Import hyperparamters

numProbLayers = sys.argv[1] #string
audioLength = 32000
numberClasses = 10

#M18-P model loaded in
_temp = __import__('loadModels.load'+numProbLayers+'pLayer', globals(), locals(), ['m'], 0)
m = _temp.m

#Testing the model, seeing what the outputs look like.
import testing

#We will write these results to a file named after the model.
modelName = "m18model_"+numProbLayers+"pLayers"
import resultsWriter

batchSizes = list(20*np.array(range(1, 25)))

#Load in the data.
import dataRawWav 
for bs in batchSizes:
    print("Starting training for batch size: " + str(bs)+"------------------------------------------")
    #We load in the data each iteration to redefine the batch size
    #We'll also split the training set for validation
    datasetVal = dataRawWav.datasetTrain.skip(hpBatch.trainNo)
    datasetTrain = dataRawWav.datasetTrain.take(hpBatch.trainNo)
    datasetTest = dataRawWav.datasetTest
    #Create batches
    batchTrain = datasetTrain.batch(bs, drop_remainder=False) #Suspect that drop_remainder=True works better
    batchVal = datasetVal.batch(bs, drop_remainder=False)
    batchTest = datasetTest.batch(bs, drop_remainder=False)

    m.load_weights('model.postCompilePreTrain') #Do this to reset the model weights dists each time
    #Set model and fit
    history = m.fit(batchTrain,
        epochs=hpBatch.epochs)
    
    print("Finished training for batch size: " + str(bs)+"------------------------------------------")

    #We test the model
    noSamples = (10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10) #All ten MC samples
    valAccuracies = testing.testMetrics(datasetVal, m, noSamples)
    testAccuracies = testing.testMetrics(datasetTest, m, noSamples)
    trainingAccuracy = history.history['accuracy'][-1]

    print(testAccuracies)
    resultsWriter.resultsWriter(valAccuracies, testAccuracies, 'Nan', 'Nan',
            modelName, 
            hpBatch.klMcSampleNo, 
            300,
            hpBatch.learningRate, 
            bs,
            hpBatch.SCALE1, 
            hpBatch.SCALE2, 
            hpBatch.PI,
            hpBatch.posterior,
            hpBatch.prior,
            hpBatch.divFn)

    for j in [i[0] for i in testAccuracies]:
        resultsWriter.resultsWriterCsvBatch(bs, j, trainingAccuracy)

    print("Finished writing results for batch size: " + str(bs)+"------------------------------------------")
