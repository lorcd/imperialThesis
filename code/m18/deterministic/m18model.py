import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models

from tensorflow.keras.layers import Flatten,MaxPooling1D,Conv1D,Softmax,GlobalAveragePooling1D
from keras.layers.core import Activation, Dense
from keras.layers.normalization import BatchNormalization
from keras import regularizers

from keras.callbacks import ReduceLROnPlateau, EarlyStopping

import numpy as np
import time

"""
This script runs the deterministic model, using other scripts in this folder to get model architecture, train and test.
"""
audioLength = 32000
numberClasses = 10

#M18 model adapted from paper. Model is defined in the following module
import loadM18Model
m = loadM18Model.m

#Next we load in the data.
#As per the paper we'll take folds 1-9 as training set and shuffle, and fix fold 10 as the test set.
#We also take a percentage of the training set out for validation during training.
#This way we can adapt the number of epochs and learning rate. This set is also used to define our "best" set of model parameter values.
import dataRawWav 

# if the accuracy does not increase over 10 epochs, we reduce the learning rate by half. Also use early stopping
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=10, min_lr=0.0001, verbose=1)
earlyStopping = EarlyStopping(monitor="val_accuracy", patience = 150)

#Testing the model, seeing what the outputs look like.
import testing

#Define model name and import results writer. 
modelName = "m18model"
import resultsWriter

#Define checkpoint callback. Based on valAccuracy, we save the best model weights and reload these back in for test time.
weights_checkpoint_filepath = 'bestM'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=weights_checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

#Train the model
t1=time.time()
history = m.fit(dataRawWav.batchTrain,
    epochs=300,
    validation_data = dataRawWav.batchVal, 
    callbacks=[model_checkpoint_callback])
print("The val accuracy attained is: ", + history.history['val_accuracy'][-1])
t2 = time.time()

#Reload in the "best" model parameters
m.load_weights(weights_checkpoint_filepath)
#Then we save the entire model(architecture, weight, everything included). This can be loaded in to load the trained model from scratch. 
m.save('wholeBestModel')

t3 = time.time()
testLoss, testAccuracy = testing.testMetrics(dataRawWav.datasetTest, m)
t4 = time.time()

#Finally, write results to file
resultsWriter.resultsWriter(testAccuracy, testLoss, modelName)
print("Time to train model(s):" + str(t2-t1))
print("Time to test model(s):" + str(t4-t3))
print("Time overall(s):" + str(t2-t1 + t4-t3))

