"""
This is the script which runs through MC sample number values, trains the model on each, and tests. All other hyperparameters are found in hpMc.py
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models

import keras.backend as K
from keras.layers import Lambda
from tensorflow.keras.layers import Flatten,MaxPooling1D,Conv1D,Softmax,GlobalAveragePooling1D
from keras.layers.core import Activation, Dense
from keras.layers.normalization import BatchNormalization
from keras import regularizers

from keras.callbacks import Callback
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.utils.np_utils import to_categorical

import cv2
import numpy as np
import sys #Use this to automate which model gets run
import hpMc #Import hyperparamters

numProbLayers = sys.argv[1] #string
audioLength = 32000
numberClasses = 10

#Load in the data.
import dataRawWav 
datasetVal = dataRawWav.datasetTrain.skip(hpMc.trainNo)
datasetTrain = dataRawWav.datasetTrain.take(hpMc.trainNo)
datasetTest = dataRawWav.datasetTest
#Create batches
batchTrain = datasetTrain.batch(hpMc.batchSize, drop_remainder=False) #Suspect that drop_remainder=True works better
batchVal = datasetVal.batch(hpMc.batchSize, drop_remainder=False)
batchTest = datasetTest.batch(hpMc.batchSize, drop_remainder=False)

#M18 model adapted from paper
_temp = __import__('loadModels.load'+numProbLayers+'pLayer', globals(), locals(), ['m'], 0)
m = _temp.m

#Testing the model, seeing what the outputs look like.
import testing

#We will write these results to a file named after the model
modelName = "m18model_"+numProbLayers+"pLayers"
import resultsWriter

#Set model and fit
history = m.fit(batchTrain,
    epochs=hpMc.epochs,
    validation_data = batchVal)

print("------------------ Following results are for M18 model with " + numProbLayers + "/18 layers probabilistic--------------------------------")
print("The val accuracy attained is: ", + history.history['val_accuracy'][-1])

#Test the trained model
noSamples = 10*tuple(range(1,30,2))
testAccuracies = testing.testMetrics(datasetTest, m, noSamples)

trainingAccuracy = history.history['accuracy'][-1]
resultsWriter.resultsWriterCsvMc(1, trainingAccuracy,"tr")

for n, j in enumerate([i[0] for i in testAccuracies]):
    resultsWriter.resultsWriterCsvMc(noSamples[n], j, "te")

