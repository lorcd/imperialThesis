"""
Here we will store all hyper parameters to keep them in one place.
This specific hyperparameter file is for the random search on scale mixture prior hyperparameters experiments.
"""

import numpy as np

epochs = 300
learningRate = 1e-4

earlyStoppingPatience = 400

batchSize = 400

klMcSampleNo = 5 #cant go over 10 for high batch size

#posterior and prior choice is also a hyper parameter (sort of)
posterior = "mean_field_logistic_fn"

prior = "custom_multivariate_laplace2_fn"
prior = "custom_multivariate_scaleMixNormalCauchy_fn"
prior = "custom_multivariate_scaleMixNormal_fn"

divFn = "divergence_fn_approx"

splitRatio = 0.9 #val/train split
#These two are set from splitRatio
trainNo = int(np.ceil(splitRatio*7895)) #number of training data points
valNo = int((1-splitRatio)*7895) #number of val data points

