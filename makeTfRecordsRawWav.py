"""
Run this script to create the tfrecords from the raw audio files. the data is vectors of length 32000.
We will make two tfRecords per fold. This brings us closest to the recommended size f a tf record as per the documentation.
"""

import tensorflow as tf
import numpy as np
import scipy.signal as sig
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
import os as os
from datetime import datetime
import librosa
import math 


folds = ["fold1","fold2","fold3","fold4","fold5","fold6","fold7","fold8","fold9","fold10"]
targetSampleRate = 8000
audioLength = 32000

"""First of all we read in a wav file. This becomes a list of floats, and we will also 
need the label which is signified by the last character of the file name.
Create a counter s.t. two hundred wav files are wrote to each tfrecord. When counter hits 200, reset and change target file.
"""

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _int64_array_feature(value):
  """Returns an int64_list from an array of  bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _float_array_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def serialize_example(feature0, label):
  """
  Creates a tf.train.Example message ready to be written to a file.
  """
  # Create a dictionary mapping the feature name to the tf.train.Example-compatible
  # data type.
  feature = {
      'feature0': _float_array_feature(feature0),
      'label': _int64_feature(label),
  }
  # Create a Features message using tf.train.Example.
  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return example_proto.SerializeToString()

yList = []
for fold in folds:
    print("Starting writing process for fold: ", fold)
    #Check fold directiory exists, if not make.
    foldPath = os.path.join("tfRecordsRawWav", fold)
    if not os.path.exists(foldPath):
        os.mkdir(foldPath)

    #Make target name
    targetFileName = os.path.join("tfRecordsRawWav", fold, "ubs.tfRecord")

    #Load in music excerpt
    audioDir = os.path.join("audio", fold) + "/"
    fileNamesAll = os.listdir(audioDir)
    fileNamesAll.remove(".DS_Store")
    counter = 0
     
    tfRecordsPerFold = 2
    samplesPerTfRec = math.ceil(len(fileNamesAll)/tfRecordsPerFold)

    #The next few lines ensure the right records are recorded to the right tfRecord
    for tfRecordNumber in range(tfRecordsPerFold):
        fileNames = fileNamesAll[tfRecordNumber*samplesPerTfRec:(tfRecordNumber + 1) * samplesPerTfRec]
        subTargetFileName = targetFileName + "_" + str(tfRecordNumber)
        print("Starting writing to " + subTargetFileName + ":  " + str(datetime.now().time()))

        with tf.io.TFRecordWriter(subTargetFileName) as writer:
            for fileName in fileNames:
                counter += 1
                
                #We take the label from the wav file name
                label = int(fileName[fileName.find("-") + 1])
                
                #Next we load in the raw wav (calling it y here, confusingly as it's really x)
                y,rate = librosa.load(audioDir + fileName, sr = targetSampleRate, mono = True)
                y = y.reshape(-1, 1)
                
                #Make sure all are the same length
                length = len(y)
                if length < audioLength:
                    y = np.concatenate((y, np.zeros(shape=(audioLength - length, 1))))
                elif length > audioLength:
                    y = y[0:audioLength]

                #Normalise to 0 mean and 1 standard deviation
                y = (y - np.mean(y)) / np.std(y)
                
                #yList.append(y) #Just a list for debugging exactly what is getting saved 

                #Finally, we write the record to the tfRecord file
                serialized_example = serialize_example(y,label)
            
                #print("Example written")
                writer.write(serialized_example)

            print("Finished writing to " + subTargetFileName + ":  " + str(datetime.now().time()) )
 
#
