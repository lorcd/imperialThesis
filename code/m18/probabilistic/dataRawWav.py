###Load in data generator which will feed to neural network
import tensorflow as tf
import numpy as np
import hp 

def getDataGen(tfrecordPaths):
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
    return dataset

tfRecordsDir = "/home/ld520/dataSets/urbanSounds/UrbanSound8K/tfRecordsRawWav/"

#To be consistent with the paper, we put folds 1-9 in the training set and 10 asthe validation
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

datasetTrain = getDataGen(trainingDataFiles) #datasetTrain contains 7895 elements, this remains unchanged
datasetTest = getDataGen(testDataFiles) #datasetTest contains 837 elements, this remains unchanged

