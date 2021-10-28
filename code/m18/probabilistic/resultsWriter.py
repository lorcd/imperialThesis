import numpy as np
import csv
import os

"""
This script stores functions used for writing results. Different functions for different experiments, as we created csvs with different headers
"""

def resultsWriter(valAccs, accs1, accs2,accs3,modelName, klMcSampleNo, epochs, lr, batchSize, scale1, scale2, pi, posterior, prior, divFn):
            """
            We use this for generally keeping track of all results across the board. We won't use for the generation of any graphs in the report.
            """
            f = open("results/testAccuracies_" + modelName + ".txt", "a+")
            f.write("\n")
            f.write("New runtime")
            f.write("\n")
            f.write("KL Divergence function sample numbers: "+str(klMcSampleNo))
            f.write("\n")
            f.write("Learning rate: "+str(lr))
            f.write("\n")
            f.write("Epochs: "+str(epochs))
            f.write("\n")
            f.write("Batch Size: "+str(batchSize))
            f.write("\n")
            f.write("Prior: " + prior)
            f.write("\n")
            f.write("Posterior: "+posterior)
            f.write("\n")
            if prior=='custom_multivariate_scaleMixNormal_fn':
                f.write("Scale 1(for scale mix prior): "+str(scale1))
                f.write("\n")
                f.write("Scale 2(for scale mix prior): "+str(scale2))
                f.write("\n")
                f.write("Pi (for scale mix prior): "+str(pi))
                f.write("\n")
            f.write("Divergence Function: "+divFn)
            f.write("\n")
            f.write("Val Accuracy (for 1, 2, 5, 10, 25, 50 MC samples): ")
            f.write("\n")
            f.write(str(valAccs))
            f.write("\n")
            f.write("Test Accuracies (for 1, 2, 5, 10, 25, 50 MC samples):")
            f.write("\n")
            f.write(str(accs1))
            f.write("\n")
            f.write(str(accs2))
            f.write("\n")
            f.write(str(accs3))
            f.write("\n")
            f.close()

def resultsWriterOverall(testAccs , trainAccs, valAccs, modelName, klMcSampleNo, epochs, lr, batchSize, scale1, scale2, pi, posterior, prior, divFn):
    #This writes results for overall models, seeking the optimal
    #First write the test accuracies
    if not os.path.isfile('results/overallTestAccuracies.csv'):
        with open('results/overallTestAccuracies.csv', 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            # write the header
            writer.writerow(["modelName", "klSampleNo", "epochs", "learningRate", "batchSize", "prior", "posterior", 'testAcc'])

    with open('results/overallTestAccuracies.csv', 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        for testAcc in testAccs:
            # write the data
            writer.writerow([modelName, klMcSampleNo, epochs, lr, batchSize, prior, posterior, testAcc])

    #Then write the accuracy histories
    if not os.path.isfile('results/overallAccuracyHists.csv'):
        with open('results/overallAccuracyHists.csv', 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            # write the header
            writer.writerow(["modelName", "klSampleNo", "epochs", "learningRate", 
                "batchSize", "prior", "posterior", 'trainAccHistory', 'valAccHistory'])

    with open('results/overallAccuracyHists.csv', 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        # write the data
        writer.writerow([modelName, klMcSampleNo, epochs, lr, batchSize, prior, posterior, trainAccs, valAccs])


def resultsWriterCsvBatch(batchSize, testAcc, trainAcc):
    if not os.path.isfile('results/batchAccuracies.csv'):
        with open('results/batchAccuracies.csv', 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            # write the header
            writer.writerow(['batchSize', 'testAcc', 'trainAcc'])

    with open('results/batchAccuracies.csv', 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)

        # write the data
        writer.writerow([batchSize, testAcc, trainAcc])

def resultsWriterCsvKl(kl, testAcc,trainAcc):
    if not os.path.isfile('results/klAccuracies.csv'):
        with open('results/klAccuracies.csv', 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            # write the header
            writer.writerow(['klSampleNumber', 'testAcc','trainAcc'])

    with open('results/klAccuracies.csv', 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)

        # write the data
        writer.writerow([kl, testAcc, trainAcc])

def resultsWriterCsvMc(mc, acc,trainOrTest):
    #Changing this function to have a train/test column, indicating whether ac is train or test.
    if not os.path.isfile('results/mcAccuracies.csv'):
        with open('results/mcAccuracies.csv', 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            # write the header
            writer.writerow(['mcSampleNumber', 'accuracy','trainOrTest'])

    with open('results/mcAccuracies.csv', 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)

        # write the data
        writer.writerow([mc, acc, trainOrTest])


def resultsWriterCsvPrior(pri, testAccs, trainAccs, valAccs):
    #Doing a little differently for priors, going to store training and val acc history too
    #First write the test accuracies
    if not os.path.isfile('results/priTestAccuracies.csv'):
        with open('results/priTestAccuracies.csv', 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            # write the header
            writer.writerow(['prior', 'testAcc'])

    with open('results/priTestAccuracies.csv', 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        for testAcc in testAccs:
            # write the data
            writer.writerow([pri, testAcc])
    
    #Then write the accuracy histories
    if not os.path.isfile('results/priAccuracyHists.csv'):
        with open('results/priAccuracyHists.csv', 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            # write the header
            writer.writerow(['prior', 'trainAccHistory', 'valAccHistory'])

    with open('results/priAccuracyHists.csv', 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        # write the data
        writer.writerow([pri, trainAccs, valAccs])

def resultsWriterCsvPriorHps(pri, scale1, scale2, pi, avgTestAcc, trainAcc, valAcc):
    #Doing a little differently for priors, going to store training and val acc history too
    #First write the test accuracies
    if not os.path.isfile('results/priHpsTestAccuracies.csv'):
        with open('results/priHpsTestAccuracies.csv', 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            # write the header
            writer.writerow(['prior', "scale1", "scale2", "pi", 'testAcc', "valAcc", "trainAcc"])

    with open('results/priHpsTestAccuracies.csv', 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([pri, scale1, scale2, pi, avgTestAcc, valAcc, trainAcc])
    
def resultsWriterCsvPosterior(post, testAccs, trainAccs, valAccs, prior):
    #Doing a little differently for posteriors, going to store training and val acc history too
    #First write the test accuracies
    if not os.path.isfile('results/postTestAccuracies.csv'):
        with open('results/postTestAccuracies.csv', 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            # write the header
            writer.writerow(['posterior', "prior", 'testAcc'])

    with open('results/postTestAccuracies.csv', 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        for testAcc in testAccs:
            # write the data
            writer.writerow([post, prior, testAcc])
    
    #Then write the accuracy histories
    if not os.path.isfile('results/postAccuracyHists.csv'):
        with open('results/postAccuracyHists.csv', 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            # write the header
            writer.writerow(['posterior', 'trainAccHistory', 'valAccHistory', 'prior'])

    with open('results/postAccuracyHists.csv', 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        # write the data
        writer.writerow([post, trainAccs, valAccs, prior])
