import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfpl = tfp.layers

from tensorflow import keras
from tensorflow.keras import models

from tensorflow.keras.layers import Flatten,MaxPooling1D,Conv1D,Softmax,GlobalAveragePooling1D
from keras.layers.core import Activation, Dense
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from keras.callbacks import Callback
from keras.callbacks import ReduceLROnPlateau

from tensorflow.keras.optimizers import RMSprop

import cv2
import numpy as np

import sys
sys.path.insert(1, '/home/ld520/git/code/m18/probabilistic') #This will need to be changed if this code is loaded in by anybody else.

audioLength = 32000
numberClasses = 10

import hp #Import hyperparameters

_temp = __import__('posteriorDists', globals(), locals(), [hp.posterior], 0)
post = _temp.__dict__[hp.posterior] #remember we call this posterior
post = post()

_temp = __import__('priorDists', globals(), locals(), [hp.prior], 0)
pri = _temp.__dict__[hp.prior]

_temp = __import__('divergenceFunctions', globals(), locals(), [hp.divFn], 0)
divergence_fn = _temp.__dict__[hp.divFn]

#M18 model adapted from paper
m = models.Sequential()

m.add(tfpl.Convolution1DReparameterization(
             64,
             input_shape=(audioLength,1), 
             kernel_size=80,
             strides=4,
             padding='same',
             kernel_prior_fn=pri,
             kernel_posterior_fn=post,
             kernel_divergence_fn = divergence_fn,
             bias_prior_fn = pri,
             bias_posterior_fn=post,
             bias_divergence_fn=divergence_fn))    
m.add(BatchNormalization())
m.add(Activation('relu'))
m.add(MaxPooling1D(pool_size=4, strides=None))

for i in range(4):
    m.add(tfpl.Convolution1DReparameterization(64,
                 kernel_size=3,
                 strides=1,
                 padding='same',
                 kernel_prior_fn=pri,
                 kernel_posterior_fn=post,
                 kernel_divergence_fn = divergence_fn,
                 bias_prior_fn = pri,
                 bias_posterior_fn=post,
                 bias_divergence_fn=divergence_fn))    
    m.add(BatchNormalization())
    m.add(Activation('relu'))
m.add(MaxPooling1D(pool_size=4, strides=None))

for i in range(4):
    m.add(tfpl.Convolution1DReparameterization(128,
                 kernel_size=3,
                 strides=1,
                 padding='same',
                 kernel_prior_fn=pri,
                 kernel_posterior_fn=post,
                 kernel_divergence_fn = divergence_fn,
                 bias_prior_fn = pri,
                 bias_posterior_fn=post,
                 bias_divergence_fn=divergence_fn))    
    m.add(BatchNormalization())
    m.add(Activation('relu'))
m.add(MaxPooling1D(pool_size=4, strides=None))

for i in range(4):
    m.add(tfpl.Convolution1DReparameterization(256,
                 kernel_size=3,
                 strides=1,
                 padding='same',
                 kernel_prior_fn=pri,
                 kernel_posterior_fn=post,
                 kernel_divergence_fn = divergence_fn,
                 bias_prior_fn = pri,
                 bias_posterior_fn=post,
                 bias_divergence_fn=divergence_fn))    
    m.add(BatchNormalization())
    m.add(Activation('relu'))
m.add(MaxPooling1D(pool_size=4, strides=None))

for i in range(4):
    m.add(tfpl.Convolution1DReparameterization(512,
                 kernel_size=3,
                 strides=1,
                 padding='same',
                 kernel_prior_fn=pri,
                 kernel_posterior_fn=post,
                 kernel_divergence_fn = divergence_fn,
                 bias_prior_fn = pri,
                 bias_posterior_fn=post,
                 bias_divergence_fn=divergence_fn))    
    m.add(BatchNormalization())
    m.add(Activation('relu'))

m.add(GlobalAveragePooling1D()) #Same as below lambda layer

m.add(tfpl.DenseReparameterization(
                        units =10,
                        activation = "softmax",
                        kernel_prior_fn=pri,
                        kernel_posterior_fn=post,
                        kernel_divergence_fn = divergence_fn,
                        bias_prior_fn = pri,
                        bias_posterior_fn=post,
                        bias_divergence_fn=divergence_fn))

#m.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=hp.learningRate),
#loss = tf.nn.sparse_softmax_cross_entropy_with_logits,
m.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp.learningRate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

m.save_weights('model.postCompilePreTrain')
#m.load_weights('model.postCompilePreTrain')

#print(m.summary())
print("------------------------------------- Model M18 loaded ---- 18 Probabilisitc layers-------------------------------------")
