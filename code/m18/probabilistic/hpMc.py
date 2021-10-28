"""
Here we will store all hyper parameters to keep them in one place.
This specific hyperparameter file is for the MC sample number experiments.
"""

import numpy as np

epochs = 300
learningRate = 1e-4

earlyStoppingPatience = 400

batchSize = 400

klMcSampleNo = 5 #cant go over 10 for high batch size

#posterior and prior choice is also a hyper parameter (sort of)
posterior = "default_mean_field_normal_fn"

prior = "custom_multivariate_scaleMixNormal_fn"

SCALE1 = np.e**3 #tried -2, -1, 0, 1,2,3
SCALE2 = np.e**-7 #tried 6, 7, 8
PI=0.5 #tried 0.25, 0.5, 0.75

divFn = "divergence_fn_approx"

splitRatio = 0.9 #val/train split
#These two are set from splitRatio
trainNo = int(np.ceil(splitRatio*7895)) #number of training data points
valNo = int((1-splitRatio)*7895) #number of val data points

