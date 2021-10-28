"""
Here we keep prior distributions for use in the probabilistic model. 
All prior distributions are loaded in from here, and the prior testing script runs through all priors defined here.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models

import tensorflow_probability as tfp
tfd = tfp.distributions
tfpl = tfp.layers
import numpy as np
import hp

#All the following are not trainable
"""
Functions are named after the distributions used in them
If a name has a number following the distribution, this stands for either a tailweight or standard variation parameter, which is varied.
For example: custom_multivariate_laplace2_fn is a prior distribution with standard deviation 2
"""
def custom_multivariate_normal_fn(dtype, shape, name, trainable, add_variable_fn):
  normal = tfd.Normal(loc = tf.zeros(shape, dtype), scale = tf.ones(shape, dtype))
  batch_ndims = tf.size(normal.batch_shape_tensor())
  return tfd.Independent(normal, reinterpreted_batch_ndims = batch_ndims)

#Cauchy Distribution
def custom_multivariate_cauchy_fn(dtype, shape, name, trainable, add_variable_fn):
  cauchy = tfd.Cauchy(loc = 0 * tf.ones(shape, dtype), scale = tf.ones(shape, dtype)) 
  batch_ndims = tf.size(cauchy.batch_shape_tensor())
  return tfd.Independent(cauchy, reinterpreted_batch_ndims = batch_ndims) 

#Exp Gamma Distribution
def custom_multivariate_expGamma_fn(dtype, shape, name, trainable, add_variable_fn):
  expGamma = tfd.ExpGamma(tf.ones(shape, dtype), rate = tf.ones(shape, dtype))
  batch_ndims = tf.size(expGamma.batch_shape_tensor())
  return tfd.Independent(expGamma, reinterpreted_batch_ndims = batch_ndims)

#Johnson SU Distribution: we will try different values of tailweight.
#The number following the main function name (xxx in custom_multivariate_johnsonSUxxx_fn) represents the tailweight value.
def custom_multivariate_johnsonSU3_fn(dtype, shape, name, trainable, add_variable_fn):
  johnsonSU = tfd.JohnsonSU(tf.zeros(shape, dtype), 3 * tf.ones(shape, dtype),tf.zeros(shape, dtype), tf.ones(shape, dtype))
  batch_ndims = tf.size(johnsonSU.batch_shape_tensor())
  return tfd.Independent(johnsonSU, reinterpreted_batch_ndims = batch_ndims)

def custom_multivariate_johnsonSU2_fn(dtype, shape, name, trainable, add_variable_fn):
  johnsonSU = tfd.JohnsonSU(tf.zeros(shape, dtype), 2 * tf.ones(shape, dtype),tf.zeros(shape, dtype), tf.ones(shape, dtype))
  batch_ndims = tf.size(johnsonSU.batch_shape_tensor())
  return tfd.Independent(johnsonSU, reinterpreted_batch_ndims = batch_ndims)

def custom_multivariate_johnsonSU1_fn(dtype, shape, name, trainable, add_variable_fn):
  johnsonSU = tfd.JohnsonSU(tf.zeros(shape, dtype), 1 * tf.ones(shape, dtype),tf.zeros(shape, dtype), tf.ones(shape, dtype))
  batch_ndims = tf.size(johnsonSU.batch_shape_tensor())
  return tfd.Independent(johnsonSU, reinterpreted_batch_ndims = batch_ndims)

def custom_multivariate_johnsonSUhalf_fn(dtype, shape, name, trainable, add_variable_fn):
  johnsonSU = tfd.JohnsonSU(tf.zeros(shape, dtype), 0.5 * tf.ones(shape, dtype),tf.zeros(shape, dtype), tf.ones(shape, dtype))
  batch_ndims = tf.size(johnsonSU.batch_shape_tensor())
  return tfd.Independent(johnsonSU, reinterpreted_batch_ndims = batch_ndims)

def custom_multivariate_johnsonSUtenth_fn(dtype, shape, name, trainable, add_variable_fn):
  johnsonSU = tfd.JohnsonSU(tf.zeros(shape, dtype), 0.1 * tf.ones(shape, dtype),tf.zeros(shape, dtype), tf.ones(shape, dtype))
  batch_ndims = tf.size(johnsonSU.batch_shape_tensor())
  return tfd.Independent(johnsonSU, reinterpreted_batch_ndims = batch_ndims)

#Laplace
def custom_multivariate_laplaceHalf_fn(dtype, shape, name, trainable, add_variable_fn):
  dist = tfd.Laplace(loc = tf.zeros(shape, dtype), scale = 0.5 * tf.ones(shape, dtype))
  batch_ndims = tf.size(dist.batch_shape_tensor())
  return tfd.Independent(dist, reinterpreted_batch_ndims = batch_ndims)

def custom_multivariate_laplace_fn(dtype, shape, name, trainable, add_variable_fn):
  dist = tfd.Laplace(loc = tf.zeros(shape, dtype), scale = tf.ones(shape, dtype))
  batch_ndims = tf.size(dist.batch_shape_tensor())
  return tfd.Independent(dist, reinterpreted_batch_ndims = batch_ndims)

def custom_multivariate_laplace2_fn(dtype, shape, name, trainable, add_variable_fn):
  dist = tfd.Laplace(loc = tf.zeros(shape, dtype), scale = 2*tf.ones(shape, dtype))
  batch_ndims = tf.size(dist.batch_shape_tensor())
  return tfd.Independent(dist, reinterpreted_batch_ndims = batch_ndims)

def custom_multivariate_logistic_fn(dtype, shape, name, trainable, add_variable_fn):
  log = tfd.Logistic(loc = tf.zeros(shape, dtype), scale = 0.5 * tf.ones(shape, dtype)) 
  batch_ndims = tf.size(log.batch_shape_tensor())
  return tfd.Independent(log, reinterpreted_batch_ndims = batch_ndims)

def custom_multivariate_weibull1_fn(dtype, shape, name, trainable, add_variable_fn): 
  weib = tfd.Weibull(tf.ones(shape, dtype), scale = 1 * tf.ones(shape, dtype)) 
  batch_ndims = tf.size(weib.batch_shape_tensor())
  return tfd.Independent(weib, reinterpreted_batch_ndims = batch_ndims)

def custom_multivariate_weibull5_fn(dtype, shape, name, trainable, add_variable_fn): 
  weib = tfd.Weibull(tf.ones(shape, dtype), scale = 5 * tf.ones(shape, dtype)) 
  batch_ndims = tf.size(weib.batch_shape_tensor())
  return tfd.Independent(weib, reinterpreted_batch_ndims = batch_ndims)

def custom_multivariate_weibull10_fn(dtype, shape, name, trainable, add_variable_fn): 
  weib = tfd.Weibull(tf.ones(shape, dtype), scale = 10 * tf.ones(shape, dtype)) 
  batch_ndims = tf.size(weib.batch_shape_tensor())
  return tfd.Independent(weib, reinterpreted_batch_ndims = batch_ndims)

def custom_multivariate_scaleMixNormal_fn(dtype, shape, name, trainable, add_variable_fn, scale1=hp.SCALE1, scale2=hp.SCALE2, pi=hp.PI):
  normal1 = tfd.Normal(loc = tf.zeros(shape, dtype), scale = scale1 * tf.ones(shape, dtype))
  normal2 = tfd.Normal(loc = tf.zeros(shape, dtype), scale = scale2 *tf.ones(shape, dtype))
  batch_ndims = tf.size(normal1.batch_shape_tensor())
  cat=tfd.Categorical(probs=[pi, 1.-pi])
  mix = tfd.Mixture(cat, 
          [tfd.Independent(normal1, reinterpreted_batch_ndims = batch_ndims), 
              tfd.Independent(normal2, reinterpreted_batch_ndims = batch_ndims)])
  return mix

def custom_multivariate_scaleMixNormalCauchy_fn(dtype, shape, name, trainable, add_variable_fn, scale1 = np.e**1, scale2 = np.e**-3, pi=0.75):
  dist1 = tfd.Normal(loc = tf.zeros(shape, dtype), scale = scale1 * tf.ones(shape, dtype))
  dist2 = tfd.Cauchy(loc = tf.zeros(shape, dtype), scale = scale2 *tf.ones(shape, dtype))
  batch_ndims = tf.size(dist1.batch_shape_tensor())
  cat=tfd.Categorical(probs=[pi, 1.-pi])
  mix = tfd.Mixture(cat, 
          [tfd.Independent(dist1, reinterpreted_batch_ndims = batch_ndims), 
              tfd.Independent(dist2, reinterpreted_batch_ndims = batch_ndims)])
  return mix

def return_custom_multivariate_scaleMixNormalCauchy_fn(scale1, scale2, pi):
    def custom_multivariate_scaleMixNormalCauchy_fn(dtype, shape, name, trainable, add_variable_fn, scale1 = scale1, scale2 = scale2, pi=pi):
      dist1 = tfd.Normal(loc = tf.zeros(shape, dtype), scale = scale1 * tf.ones(shape, dtype))
      dist2 = tfd.Cauchy(loc = tf.zeros(shape, dtype), scale = scale2 *tf.ones(shape, dtype))
      batch_ndims = tf.size(dist1.batch_shape_tensor())
      cat=tfd.Categorical(probs=[pi, 1.-pi])
      mix = tfd.Mixture(cat, 
              [tfd.Independent(dist1, reinterpreted_batch_ndims = batch_ndims), 
                  tfd.Independent(dist2, reinterpreted_batch_ndims = batch_ndims)])
      return mix
    return custom_multivariate_scaleMixNormalCauchy_fn

def return_custom_multivariate_scaleMixNormal_fn(scale1, scale2, pi):
    def custom_multivariate_scaleMixNormalCauchy_fn(dtype, shape, name, trainable, add_variable_fn, scale1 = scale1, scale2 = scale2, pi=pi):
      dist1 = tfd.Normal(loc = tf.zeros(shape, dtype), scale = scale1 * tf.ones(shape, dtype))
      dist2 = tfd.Normal(loc = tf.zeros(shape, dtype), scale = scale2 *tf.ones(shape, dtype))
      batch_ndims = tf.size(dist1.batch_shape_tensor())
      cat=tfd.Categorical(probs=[pi, 1.-pi])
      mix = tfd.Mixture(cat, 
              [tfd.Independent(dist1, reinterpreted_batch_ndims = batch_ndims), 
                  tfd.Independent(dist2, reinterpreted_batch_ndims = batch_ndims)])
      return mix
    return custom_multivariate_scaleMixNormal_fn
