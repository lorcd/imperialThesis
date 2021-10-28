"""
Here we keep posterior distributions for use
in the probabilistic model. All posteriors are loaded in from here, including the posterior tester script which runs through all defined here.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models

import tensorflow_probability as tfp
tfd = tfp.distributions
tfpl = tfp.layers

#take the source code for the default posterior
# These are trainable
#This is same as default. Will use this as a template to create other posts
def mean_field_normal_fn(
    is_singular=False,
    loc_initializer=tf.initializers.random_normal(stddev=0.1),
    untransformed_scale_initializer=tf.initializers.random_normal(
        mean=-3., stddev=0.1),
    loc_regularizer=None,
    untransformed_scale_regularizer=None,
    loc_constraint=None,
    untransformed_scale_constraint=None):

  loc_scale_fn = tfpl.default_loc_scale_fn(
      is_singular=is_singular,
      loc_initializer=loc_initializer,
      untransformed_scale_initializer=untransformed_scale_initializer,
      loc_regularizer=loc_regularizer,
      untransformed_scale_regularizer=untransformed_scale_regularizer,
      loc_constraint=loc_constraint,
      untransformed_scale_constraint=untransformed_scale_constraint)
  def _fn(dtype, shape, name, trainable, add_variable_fn):
    loc, scale = loc_scale_fn(dtype, shape, name, trainable, add_variable_fn)
    if scale is None:
      dist = tfd.Deterministic(loc=loc)
    else:
      dist = tfd.Normal(loc=loc, scale=scale)
    batch_ndims = tf.size(dist.batch_shape_tensor())
    return tfd.Independent(
        dist, reinterpreted_batch_ndims=batch_ndims)
  return _fn

def mean_field_cauchy_fn(
    is_singular=False,
    loc_initializer=tf.initializers.random_normal(stddev=0.1),
    untransformed_scale_initializer=tf.initializers.random_normal(
        mean=-3., stddev=0.1),
    loc_regularizer=None,
    untransformed_scale_regularizer=None,
    loc_constraint=None,
    untransformed_scale_constraint=None):

  loc_scale_fn = tfpl.default_loc_scale_fn(
      is_singular=is_singular,
      loc_initializer=loc_initializer,
      untransformed_scale_initializer=untransformed_scale_initializer,
      loc_regularizer=loc_regularizer,
      untransformed_scale_regularizer=untransformed_scale_regularizer,
      loc_constraint=loc_constraint,
      untransformed_scale_constraint=untransformed_scale_constraint)
  def _fn(dtype, shape, name, trainable, add_variable_fn):
    loc, scale = loc_scale_fn(dtype, shape, name, trainable, add_variable_fn)
    if scale is None:
      dist = tfd.Deterministic(loc=loc)
    else:
      dist = tfd.Cauchy(loc=loc, scale=scale)
    batch_ndims = tf.size(dist.batch_shape_tensor())
    return tfd.Independent(
        dist, reinterpreted_batch_ndims=batch_ndims)
  return _fn

#more similar to the normal distn
def mean_field_logistic_fn(
    is_singular=False,
    loc_initializer=tf.initializers.random_normal(stddev=0.1),
    untransformed_scale_initializer=tf.initializers.random_normal(
        mean=-3., stddev=0.1),
    loc_regularizer=None,
    untransformed_scale_regularizer=None,
    loc_constraint=None,
    untransformed_scale_constraint=None):

  loc_scale_fn = tfpl.default_loc_scale_fn(
      is_singular=is_singular,
      loc_initializer=loc_initializer,
      untransformed_scale_initializer=untransformed_scale_initializer,
      loc_regularizer=loc_regularizer,
      untransformed_scale_regularizer=untransformed_scale_regularizer,
      loc_constraint=loc_constraint,
      untransformed_scale_constraint=untransformed_scale_constraint)
  def _fn(dtype, shape, name, trainable, add_variable_fn):
    loc, scale = loc_scale_fn(dtype, shape, name, trainable, add_variable_fn)
    if scale is None:
      dist = tfd.Deterministic(loc=loc)
    else:
      dist = tfd.Logistic(loc=loc, scale=scale)
    batch_ndims = tf.size(dist.batch_shape_tensor())
    return tfd.Independent(
        dist, reinterpreted_batch_ndims=batch_ndims)
  return _fn

def mean_field_laplace_fn(
    is_singular=False,
    loc_initializer=tf.initializers.random_normal(stddev=0.1),
    untransformed_scale_initializer=tf.initializers.random_normal(
        mean=-3., stddev=0.1),
    loc_regularizer=None,
    untransformed_scale_regularizer=None,
    loc_constraint=None,
    untransformed_scale_constraint=None):

  loc_scale_fn = tfpl.default_loc_scale_fn(
      is_singular=is_singular,
      loc_initializer=loc_initializer,
      untransformed_scale_initializer=untransformed_scale_initializer,
      loc_regularizer=loc_regularizer,
      untransformed_scale_regularizer=untransformed_scale_regularizer,
      loc_constraint=loc_constraint,
      untransformed_scale_constraint=untransformed_scale_constraint)
  def _fn(dtype, shape, name, trainable, add_variable_fn):
    loc, scale = loc_scale_fn(dtype, shape, name, trainable, add_variable_fn)
    if scale is None:
      dist = tfd.Deterministic(loc=loc)
    else:
      dist = tfd.Laplace(loc=loc, scale=scale)
    batch_ndims = tf.size(dist.batch_shape_tensor())
    return tfd.Independent(
        dist, reinterpreted_batch_ndims=batch_ndims)
  return _fn

def mean_field_studentT6_fn(
    is_singular=False,
    loc_initializer=tf.initializers.random_normal(stddev=0.1),
    untransformed_scale_initializer=tf.initializers.random_normal(
        mean=-3., stddev=0.1),
    loc_regularizer=None,
    untransformed_scale_regularizer=None,
    loc_constraint=None,
    untransformed_scale_constraint=None):

  loc_scale_fn = tfpl.default_loc_scale_fn(
      is_singular=is_singular,
      loc_initializer=loc_initializer,
      untransformed_scale_initializer=untransformed_scale_initializer,
      loc_regularizer=loc_regularizer,
      untransformed_scale_regularizer=untransformed_scale_regularizer,
      loc_constraint=loc_constraint,
      untransformed_scale_constraint=untransformed_scale_constraint)
  def _fn(dtype, shape, name, trainable, add_variable_fn):
    loc, scale = loc_scale_fn(dtype, shape, name, trainable, add_variable_fn)
    if scale is None:
      dist = tfd.Deterministic(loc=loc)
    else:
      dist = tfd.StudentT(df=6, loc=loc, scale=scale)
    batch_ndims = tf.size(dist.batch_shape_tensor())
    return tfd.Independent(
        dist, reinterpreted_batch_ndims=batch_ndims)
  return _fn

def mean_field_studentT3_fn(
    is_singular=False,
    loc_initializer=tf.initializers.random_normal(stddev=0.1),
    untransformed_scale_initializer=tf.initializers.random_normal(
        mean=-3., stddev=0.1),
    loc_regularizer=None,
    untransformed_scale_regularizer=None,
    loc_constraint=None,
    untransformed_scale_constraint=None):

  loc_scale_fn = tfpl.default_loc_scale_fn(
      is_singular=is_singular,
      loc_initializer=loc_initializer,
      untransformed_scale_initializer=untransformed_scale_initializer,
      loc_regularizer=loc_regularizer,
      untransformed_scale_regularizer=untransformed_scale_regularizer,
      loc_constraint=loc_constraint,
      untransformed_scale_constraint=untransformed_scale_constraint)
  def _fn(dtype, shape, name, trainable, add_variable_fn):
    loc, scale = loc_scale_fn(dtype, shape, name, trainable, add_variable_fn)
    if scale is None:
      dist = tfd.Deterministic(loc=loc)
    else:
      dist = tfd.StudentT(df=3, loc=loc, scale=scale)
    batch_ndims = tf.size(dist.batch_shape_tensor())
    return tfd.Independent(
        dist, reinterpreted_batch_ndims=batch_ndims)
  return _fn

def mean_field_studentT1_fn(
    is_singular=False,
    loc_initializer=tf.initializers.random_normal(stddev=0.1),
    untransformed_scale_initializer=tf.initializers.random_normal(
        mean=-3., stddev=0.1),
    loc_regularizer=None,
    untransformed_scale_regularizer=None,
    loc_constraint=None,
    untransformed_scale_constraint=None):

  loc_scale_fn = tfpl.default_loc_scale_fn(
      is_singular=is_singular,
      loc_initializer=loc_initializer,
      untransformed_scale_initializer=untransformed_scale_initializer,
      loc_regularizer=loc_regularizer,
      untransformed_scale_regularizer=untransformed_scale_regularizer,
      loc_constraint=loc_constraint,
      untransformed_scale_constraint=untransformed_scale_constraint)
  def _fn(dtype, shape, name, trainable, add_variable_fn):
    loc, scale = loc_scale_fn(dtype, shape, name, trainable, add_variable_fn)
    if scale is None:
      dist = tfd.Deterministic(loc=loc)
    else:
      dist = tfd.StudentT(df=1, loc=loc, scale=scale)
    batch_ndims = tf.size(dist.batch_shape_tensor())
    return tfd.Independent(
        dist, reinterpreted_batch_ndims=batch_ndims)
  return _fn

def mean_field_johnsonSU_fn(
    is_singular=False,
    loc_initializer=tf.initializers.random_normal(stddev=0.1),
    untransformed_scale_initializer=tf.initializers.random_normal(
        mean=-3., stddev=0.1),
    loc_regularizer=None,
    untransformed_scale_regularizer=None,
    loc_constraint=None,
    untransformed_scale_constraint=None):

  loc_scale_fn = tfpl.default_loc_scale_fn(
      is_singular=is_singular,
      loc_initializer=loc_initializer,
      untransformed_scale_initializer=untransformed_scale_initializer,
      loc_regularizer=loc_regularizer,
      untransformed_scale_regularizer=untransformed_scale_regularizer,
      loc_constraint=loc_constraint,
      untransformed_scale_constraint=untransformed_scale_constraint)
  def _fn(dtype, shape, name, trainable, add_variable_fn):
    loc, scale = loc_scale_fn(dtype, shape, name, trainable, add_variable_fn)
    if scale is None:
      dist = tfd.Deterministic(loc=loc)
    else:
      dist = tfd.JohnsonSU(skewness=0, tailweight=1, loc=loc, scale=scale)
    batch_ndims = tf.size(dist.batch_shape_tensor())
    return tfd.Independent(
        dist, reinterpreted_batch_ndims=batch_ndims)
  return _fn


