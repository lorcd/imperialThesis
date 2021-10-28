"""
This is the script which runs through kl divergence sample number values, trains the model on each, and tests. All other hyperparameters are found in hpKl.py
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
import hpKl #Import hyperparamters

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
datasetVal = dataRawWav.datasetTrain.skip(hpKl.trainNo)
datasetTrain = dataRawWav.datasetTrain.take(hpKl.trainNo)
datasetTest = dataRawWav.datasetTest
#Create batches
batchTrain = datasetTrain.batch(hpKl.batchSize, drop_remainder=False) #Suspect that drop_remainder=True works better
batchVal = datasetVal.batch(hpKl.batchSize, drop_remainder=False)
batchTest = datasetTest.batch(hpKl.batchSize, drop_remainder=False)

klDivSamples = list(np.array(range(10, 20)))

for klDivSample in klDivSamples:
    print("Starting training for KL Divergence approximation samples: " + str(klDivSample)+"------------------------------------------")
    #Define the divergnce function with the kl div sample number
    def kl_approx(q, p, _, n=klDivSample):
        q_tensor = q.sample(n)
        return tf.reduce_mean(q.log_prob(q_tensor) - p.log_prob(q_tensor))

    divergence_fn_approx = lambda q, p, _ : kl_approx(q, p, _) / hpKl.trainNo
    
    #This function ceates a model based on the divergence function and returns it.
    import loadModels.load18pLayerKlFunction
    m = loadModels.load18pLayerKlFunction.returnModel(divergence_fn_approx)
    #Might work without this.m.load_weights('model.postCompilePreTrain') #Do this to reset the model weights dists each time
    
    #Set model and fit
    history = m.fit(batchTrain,
        epochs=hpKl.epochs)
    #validation_data = batchVal)
    
    print("Finished training for KL Divergence approximation samples: " + str(klDivSample)+"------------------------------------------")

    #Testing this one is a little different because we can take several predictions and average
    noSamples = (10, 10, 10, 10, 10) #All ten MC samples
    valAccuracies = testing.testMetrics(datasetVal, m, noSamples)
    testAccuracies = testing.testMetrics(datasetTest, m, noSamples)
    trainingAccuracy = history.history['accuracy'][-1]
    print(testAccuracies)

    resultsWriter.resultsWriter(valAccuracies, testAccuracies, 'Nan', 'Nan',
            modelName, 
            klDivSample, 
            hpKl.epochs,
            hpKl.learningRate, 
            hpKl.batchSize,
            #hpKl.earlyStoppingPatience, 
            hpKl.SCALE1, 
            hpKl.SCALE2, 
            hpKl.PI,
            hpKl.posterior,
            hpKl.prior,
            hpKl.divFn)

    #medianValAcc = np.median([i[0] for i in valAccuracies])
    #medianTestAcc = np.median([i[0] for i in testAccuracies])
    for j in [i[0] for i in testAccuracies]:
        resultsWriter.resultsWriterCsvKl(klDivSample, j, trainingAccuracy)

    print("Finished writing results for KL Divergence approximation samples: " + str(klDivSample)+"------------------------------------------")
