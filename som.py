"""
 -----------------------------------------------------------------------
|                                                                       |
|   Class:          SelfOrganizingMap                                   |
|   Description:    Implementation of a self organizing map.            |
|                                                                       |
|                                                                       |
|                                                                       |
|   Authors:        Zayd Bille                                          |
|   Date:           11/14/2017                                          |
|                                                                       |
 -----------------------------------------------------------------------
"""

from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt
from random import randint
from numpy import arange
import numpy as np
import random

class SelfOrganizingMap:
    
    """
     ----------------------------------------------------------------
    |                      ---------------                           |
    |                     | Class Members |                          |
    |                       ---------------                          |
    |                                                                |
    |      int trainSize      - Amount of data designated for        |
    |                           training/classification.             |
    |                                                                |
    |       int testSize      - Amount of data designated for        |
    |                           potential testing (10,000            |
    |                           for MNIST).                          |
    |                                                                |
    |        int epochs       - A single epoch is one pass through   |
    |                           the whole training dataset.          |
    |                                                                |
    |     int learningRate    - Learning rate.                       |
    |                                                                |
    |    int neuronGridSize   - Size of the classification grid.     |
    |                           (neuronGridSize x neuronGridSize).   |
    |                                                                |
    |        matrix som       - A weight matrix that represents      |
    |                           the actual self-organizing map.      |
    |                                                                |
     ----------------------------------------------------------------
    """
    
    def __init__(self, 
                 trainSize = 60000,
                 testSize = 10000,
                 epochs = 2000,
                 learningRate = 0.01,
                 neuronGridSize = 20):
        self.trainSize = trainSize
        self.testSize = testSize
        self.epochs = epochs
        self.learningRate = learningRate
        self.neuronGridSize = neuronGridSize
        self.som = None
        self.visualSom = None

    def loadTrainData(self):
        mnist = fetch_mldata('MNIST original')
        
        sets = arange(len(mnist.data))
        
        trainIndex = arange(0, self.trainSize)

        images, labels = mnist.data[trainIndex], mnist.target[trainIndex]
        
        return images, labels

    def loadTestData(self):
        mnist = fetch_mldata('MNIST original')
        
        sets = arange(len(mnist.data))

        testIndex = arange(self.trainSize + 1, self.trainSize + self.testSize)

        images, labels = mnist.data[testIndex], mnist.target[testIndex]
        
        return images, labels
        
    def createSom(self, emptyGrid, vd):
        #self.som = np.random.random((emptyGrid[0], emptyGrid[1], vd))
        self.som = np.random.random_integers(0, 255, (emptyGrid[0], emptyGrid[1], 784))
        self.visualSom = np.random.random((emptyGrid[0], emptyGrid[1]))
        for i in range(self.visualSom.shape[0]):
            for j in range(self.visualSom.shape[1]):
                self.visualSom[i, j] = random.choice([1, 1])
        
    def findWinnerNeuron(self, anImage, m):
        winnerIndex = np.array([0, 0])
        
        ## Start with a really small number. ##
        minimum = np.iinfo(np.int).max
        
        ## Determine the best neuron. ##
        for i in range(self.som.shape[0]):
            for j in range(self.som.shape[1]):
                w = self.som[i, j, :].reshape(m, 1)
                distance = np.sum((w - anImage) ** 2)
                if distance < minimum:
                    minimum = distance
                    winnerIndex = np.array([i, j])
                    
        ## Get neuron at winnerIndex. ##
        winner = self.som[winnerIndex[0], winnerIndex[1], :].reshape(m, 1)
        return (winner, winnerIndex)
        
    def classify(self):
        ##############
        ### Step 1 ###
        ##############
        ## Load the training data and extract its dimensions. ##
        images, labels = self.loadTrainData()
        m = images.shape[0]
        n = images.shape[1]
        
        ## Filter out numbers other than the 1's and 5's ##
        # images, labels = zip(*[(i, j) for i, j in zip(images, labels) if j == 1 or j == 5])
        
        ## Use the the dimension of the data to create the network. ##
        emptyGrid = np.array([self.neuronGridSize, self.neuronGridSize])
        self.createSom(emptyGrid, n)

        ## Area of influence of each neuron in the grid. ##
        influenceArea = max(emptyGrid[0], emptyGrid[1]) / 2
        
        ## Influence decays over time. ##
        decayConstant = self.epochs / np.log(influenceArea)
        
        ## Save the initial learning rate. ##
        learningRate = self.learningRate
        
        ##############
        ### Step 2 ###
        ##############
        normalise_data = True

        # if True, assume all data is on common scale
        # if False, normalise to [0 1] range along each column
        normalise_by_column = False

        # we want to keep a copy of the raw data for later
        data = images

        # check if data needs to be normalised
        if normalise_data:
            if normalise_by_column:
                # normalise along each column
                col_maxes = images.max(axis=0)
                data = images / col_maxes[np.newaxis, :]
            else:
                # normalise entire dataset
                data = images / data.max()
                
        ##############
        ### Step 3 ###
        ##############
        for i in range(self.epochs):
            print("---I'm classifying ... Iteration: %d/%d" % (i + 1, self.epochs))            
            
            ## Randomly choose an image in the training set. ##

            #anImage = data[:, np.random.randint(0, m)].reshape(np.array([n, 1]))
            
            # ensure that the image is either a 1 or 5
            randomIndex = randint(0, m-1)
            if (labels[randomIndex] != 1 or labels[randomIndex] != 5):
                while (labels[randomIndex] != 1 and labels[randomIndex] != 5):
                    # keep looking until it's either a 1 or 5 (super inefficient but whatever)
                    randomIndex = randint(0, m-1)
            anImage = images[randomIndex].reshape(n, 1)
            aLabel = labels[randomIndex]
                        
            ## Find the winner neuron. ##
            winner, winnerIndex = self.findWinnerNeuron(anImage, n)
            
            ## Adjust the learning rate and the influence area by decay. ##
            influenceArea = influenceArea * np.exp(-i / decayConstant)
            learningRate = learningRate * np.exp(-i / self.epochs)
            
            ## With the winner neuron, update it to shift closer to a ##
            ## a certain cluster and then move all of its neighbours  ##
            ## closer to that cluster by a reduced amount.            ##
            for i in range(self.som.shape[0]):
                for j in range(self.som.shape[1]):
                    neuron = self.som[i, j, :].reshape(n, 1)
                    
                    ## Get the distance for this neuron. ##
                    neuronDistance = np.sum((np.array([i, j]) - winnerIndex) ** 2)
                    
                    ## Determine if it's within the influence area. ##
                    if neuronDistance <= influenceArea**2:
                        ## Determine how much it is influenced by. ##
                        influence = np.exp(-neuronDistance / (2* (influenceArea**2)))
                        
                        ## Update the neuron. ##
                        newNeuron = neuron + (learningRate * influence * (anImage - neuron))
                        self.som[i, j, :] = newNeuron.reshape(1, n)
                        self.visualSom[i, j] = aLabel
    
    def classifyWithKMeans(self):
        return 0
"""
 ----------------------------------------------------------------
|    main()                                                      |
|    ------                                                      |
|                                                                |
|    Where the magic happens.                                    |
 ----------------------------------------------------------------
"""
def main():
    som = SelfOrganizingMap(
                trainSize = 60000,
                testSize = 10000,
                epochs = 20,
                learningRate = 0.01,
                neuronGridSize = 100)
      
    som.classify()
    
    plt.imshow(som.visualSom, cmap='hot', interpolation='nearest')
    plt.grid(b=None, which='major', axis='both')
    plt.show()
    
    

main()
