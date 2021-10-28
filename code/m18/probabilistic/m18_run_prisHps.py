"""
This is the script which runs through scale mixture prior hyperparameters (scale 1, scale 2, mixture ratio), trains the model on each, and tests. All other hyperparameters are found in hpPrisHps.py
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
import hpPrisHps #Import hyperparamters

numProbLayers = sys.argv[1] 
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
datasetVal = dataRawWav.datasetTrain.skip(hpPrisHps.trainNo)
datasetTrain = dataRawWav.datasetTrain.take(hpPrisHps.trainNo)
datasetTest = dataRawWav.datasetTest
#Create batches
batchTrain = datasetTrain.batch(hpPrisHps.batchSize, drop_remainder=False) #Suspect that drop_remainder=True works better
batchVal = datasetVal.batch(hpPrisHps.batchSize, drop_remainder=False)
batchTest = datasetTest.batch(hpPrisHps.batchSize, drop_remainder=False)

priorModule = __import__('priorDists', globals(), locals(), [], 0)
priorStr = hpPrisHps.prior

piPossibles = [0.25, 0.5, 0.75]

#Try random search over parameters
def randomBetween(x, y):
    #x is smaller
    return x + np.random.rand()*(y + x)

nn = 3
scale1list = [randomBetween(np.e**-3, np.e**3) for i in range(nn)] #three randoms between e^-3 and e^3
scale2list = [randomBetween(np.e**-9, np.e**-3) for i in range(nn)] #three randoms between e^-3 and e^3
piList = [np.random.choice(piPossibles) for i in range(nn)]

cnt=0

for pi in piPossibles:
    for scale1 in scale1list:
        for scale2 in scale2list:
                cnt+=1
                print("Starting training for prior: " + str(cnt)+"/"+str(nn*3)+"------------------------------------------")
                #Define the prior function
                pri = priorModule.__dict__["return_" + priorStr](scale1=scale1, scale2=scale2, pi=pi)

                #This function ceates a model based on the inputted prior and returns it.
                import loadModels.load18pLayerPriorFunction
                m = loadModels.load18pLayerPriorFunction.returnModel(pri)
                
                #Set model and fit
                history = m.fit(batchTrain,
                    epochs=hpPrisHps.epochs,
                    validation_data = batchVal)
                
                print("Finished training for prior: " + str(cnt)+"/"+str(nn*3)+"------------------------------------------")
                
                #Testing
                noSamples = (10, 10, 10, 10, 10) #All ten MC samples
                valAccuracies = testing.testMetrics(datasetVal, m, noSamples)
                testAccuracies = testing.testMetrics(datasetTest, m, noSamples)
                avgTestAccuracy = np.mean(testAccuracies)
                trainAccuracyFinal = history.history['accuracy'][-1]
                trainAccuracyHist = history.history['accuracy']
                valAccuracyHist = history.history['val_accuracy']
                valAccuracyFinal = history.history['val_accuracy'][-1]
                print(testAccuracies)
                
                resultsWriter.resultsWriterCsvPriorHps(priorStr, scale1, scale2, pi, avgTestAccuracy, trainAccuracyFinal, valAccuracyFinal)
                print("Finished writing results for a set of prior hps: " + str(cnt)+"/"+str(nn*3)+"------------------------------------------")
