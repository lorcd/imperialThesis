"""
This is the script which runs M18-P once, trains the model and tests. All other hyperparameters are found in hp.py. 
It also saves the best model as per val accuracy.
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models

import cv2
import numpy as np
import sys #Use this to automate which model gets run
import hp #Import hyperparamters
import time

numProbLayers = sys.argv[1] #string
audioLength = 32000
numberClasses = 10

#Load in the data.
import dataRawWav 
datasetVal = dataRawWav.datasetTrain.skip(hp.trainNo)
datasetTrain = dataRawWav.datasetTrain.take(hp.trainNo)
datasetTest = dataRawWav.datasetTest
#Create batches
batchTrain = datasetTrain.batch(hp.batchSize, drop_remainder=False) 
batchVal = datasetVal.batch(hp.batchSize, drop_remainder=False)
batchTest = datasetTest.batch(hp.batchSize, drop_remainder=False)

#M18 model adapted from paper
_temp = __import__('loadModels.load'+numProbLayers+'pLayer', globals(), locals(), ['m'], 0)
m = _temp.m

#Model chackpoint should be used
weights_checkpoint_filepath = 'bestM'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=weights_checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

#Testing the model, seeing what the outputs look like.
import testing

#We will write these results to a file named after the model
modelName = "m18model_"+numProbLayers+"pLayers"
import resultsWriter

#Set model and fit
t1 = time.time()
history = m.fit(batchTrain,
    epochs=hp.epochs,
    validation_data = batchVal,
    callbacks = [model_checkpoint_callback])
t2 = time.time()

m.load_weights(weights_checkpoint_filepath)
m.save('wholeBestModel')

print("------------------ Following results are for M18 model with " + numProbLayers + "/18 layers probabilistic--------------------------------")
print("The val accuracy attained is: ", + history.history['val_accuracy'][-1])

noSamples = 20*(50,)

#This gets 20 predictions each of 50 MC samples each. We calculate the time taken and divide by twenty for one.
t3 = time.time()
testAccuracies = testing.testMetrics(datasetTest, m, noSamples)
t4 = time.time()

trainAccuracyHist = history.history['accuracy']
valAccuracyHist = history.history['val_accuracy']

print(testAccuracies)
resultsWriter.resultsWriterOverall(testAccuracies,trainAccuracyHist, valAccuracyHist,
        modelName, 
        hp.klMcSampleNo, 
        hp.epochs,
        hp.learningRate, 
        hp.batchSize,
        hp.SCALE1, 
        hp.SCALE2, 
        hp.PI,
        hp.posterior,
        hp.prior,
        hp.divFn)

print("Time to train model(s):" + str(t2-t1))
print("Time to test model(s):" + str((t4-t3)/len(noSamples)))
print("Time overall(s):" + str(t2-t1 + (t4-t3)/len(noSamples)))

