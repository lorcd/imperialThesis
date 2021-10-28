import numpy as np
import os

def resultsWriter(acc, loss, modelName):
            if not os.path.isfile("results/testAccuracies_" + modelName + ".txt"):
                with open("results/testAccuracies_" + modelName + ".txt", 'w', encoding='UTF8', newline='') as f:
                    # write the header
                    f.write('testAcc')

            if not os.path.isfile("results/testLosses_" + modelName + ".txt"):
                with open("results/testLosses_" + modelName + ".txt", 'w', encoding='UTF8', newline='') as f:
                    # write the header
                    f.write('testLoss')

            f = open("results/testAccuracies_" + modelName + ".txt", "a+")
            f.write('\n')
            f.write(str(acc))
            f.close()

            g = open("results/testLosses_" + modelName + ".txt", "a+")
            g.write('\n')
            g.write(str(loss))
            g.close()

