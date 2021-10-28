###Load in data generator which will feed to neural network
import tensorflow as tf
import numpy as np

"""
Here we load in the data. This is called in the main script. Batch size is defined here
"""

def getDataGen(tfrecordPaths, batchSize=40):
    def decode(serialized_example):
            features = {
                "feature0": tf.io.FixedLenFeature([32000,1], tf.float32),
                "label": tf.io.FixedLenFeature([], tf.int64),
         }
            example = tf.io.parse_single_example(serialized_example, features)

            feature0 = example['feature0']
            label = example['label']
            return (feature0,
                    label)
    
    dataset = tf.data.TFRecordDataset(tfrecordPaths)
    dataset = dataset.map(decode) 
    dataset = dataset.shuffle(8000) #This ensures all data is shuffled before being outputted
    #batch = batch.prefetch(2)
    
    #iterator = iter(dataset)
    #Feeds into with print(iterator.get_next())
    return dataset

batchSize = 128
tfRecordsDir = "/home/ld520/dataSets/urbanSounds/UrbanSound8K/tfRecordsRawWav/"

#To be consistent with the paper, we put folds 1-9 in the training set and 10 as the test
trainingDataFiles = (tfRecordsDir+"fold1/ubs.tfRecord_0",tfRecordsDir+"fold1/ubs.tfRecord_1",
        tfRecordsDir+"fold2/ubs.tfRecord_0",tfRecordsDir+"fold2/ubs.tfRecord_1",
        tfRecordsDir+"fold3/ubs.tfRecord_0",tfRecordsDir+"fold3/ubs.tfRecord_1",
        tfRecordsDir+"fold4/ubs.tfRecord_0",tfRecordsDir+"fold4/ubs.tfRecord_1",
        tfRecordsDir+"fold5/ubs.tfRecord_0",tfRecordsDir+"fold5/ubs.tfRecord_1",
        tfRecordsDir+"fold6/ubs.tfRecord_0",tfRecordsDir+"fold6/ubs.tfRecord_1",
        tfRecordsDir+"fold7/ubs.tfRecord_0",tfRecordsDir+"fold7/ubs.tfRecord_1",
        tfRecordsDir+"fold8/ubs.tfRecord_0",tfRecordsDir+"fold8/ubs.tfRecord_1",
        tfRecordsDir+"fold9/ubs.tfRecord_0",tfRecordsDir+"fold9/ubs.tfRecord_1")

testDataFiles = (tfRecordsDir+"fold10/ubs.tfRecord_0",tfRecordsDir+"fold10/ubs.tfRecord_1")

datasetTrain = getDataGen(trainingDataFiles, batchSize = batchSize) #datasetTrain contains 7895 elements, this remains unchanged
datasetTest = getDataGen(testDataFiles, batchSize = batchSize) #datasetTest contains 837 elements, this remains unchanged

#We'll also split the training set for validation
splitRatio = 0.9
trainNo = int(np.ceil(splitRatio*7895)) 
valNo = int((1-splitRatio)*7895)

datasetVal = datasetTrain.skip(trainNo)
datasetTrain = datasetTrain.take(trainNo)

#Create batches
batchTrain = datasetTrain.batch(batchSize, drop_remainder=False) #Suspect that drop_remainder=True works better
batchVal = datasetVal.batch(batchSize, drop_remainder=False)
batchTest = datasetTest.batch(batchSize, drop_remainder=False)

