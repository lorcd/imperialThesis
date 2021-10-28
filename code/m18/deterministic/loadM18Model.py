import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models

from tensorflow.keras.layers import Flatten,MaxPooling1D,Conv1D,Softmax,GlobalAveragePooling1D
from keras.layers.core import Activation, Dense
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from keras.callbacks import ReduceLROnPlateau

import cv2
import numpy as np

"""
This script just loads in the deterministic model
"""

audioLength = 32000
numberClasses = 10

#M18 model adapted from paper
m = models.Sequential()

m.add(Conv1D(64,
             input_shape=(audioLength,1),
             kernel_size=80,
             strides=4,
             padding='same',
             kernel_initializer='glorot_uniform',
             kernel_regularizer=regularizers.l2(l=0.0001)))
m.add(BatchNormalization())
m.add(Activation('relu'))
m.add(MaxPooling1D(pool_size=4, strides=None))

for i in range(4):
    m.add(Conv1D(64,
                 kernel_size=3,
                 strides=1,
                 padding='same',
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=regularizers.l2(l=0.0001)))
    m.add(BatchNormalization())
    m.add(Activation('relu'))
m.add(MaxPooling1D(pool_size=4, strides=None))

for i in range(4):
    m.add(Conv1D(128,
                 kernel_size=3,
                 strides=1,
                 padding='same',
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=regularizers.l2(l=0.0001)))
    m.add(BatchNormalization())
    m.add(Activation('relu'))
m.add(MaxPooling1D(pool_size=4, strides=None))

for i in range(4):
    m.add(Conv1D(256,
                 kernel_size=3,
                 strides=1,
                 padding='same',
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=regularizers.l2(l=0.0001)))
    m.add(BatchNormalization())
    m.add(Activation('relu'))
m.add(MaxPooling1D(pool_size=4, strides=None))

for i in range(4):
    m.add(Conv1D(512,
                 kernel_size=3,
                 strides=1,
                 padding='same',
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=regularizers.l2(l=0.0001)))
    m.add(BatchNormalization())
    m.add(Activation('relu'))

m.add(GlobalAveragePooling1D()) 
m.add(Dense(10, activation = "softmax"))

m.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

print(m.summary())
