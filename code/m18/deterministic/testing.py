#Testing the model, seeing what the outputs look like.
import tensorflow as tf
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import numpy as np

def testMetrics(dataSet, model):
    wavForms, labels = list(dataSet.batch(1000))[0] #We just take 1000 cuz it's bigger than test set size
    predictions = model(wavForms, training=False)
    
    #CE Loss
    scce = SparseCategoricalCrossentropy()
    loss = scce(labels, predictions).numpy()

    #Accuracy
    predictionsInt = np.argmax(predictions.numpy(), axis=1)
    comp = predictionsInt == labels.numpy()
    accuracy = sum(comp)/len(comp)
    print("Test Loss: " + str(loss) + ". Test accuracy: " + str(accuracy))

    return loss, accuracy






