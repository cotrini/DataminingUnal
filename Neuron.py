import numpy as np

class Neuron:
   
    def __init__(self):
        self.weights = []
        self.inputs = []
        self.value = 0
        self.bias = 0
        self.error = 1
        self.output = 0
        self.totalWeights = 0

    def activationFunction(self):
        self.output = (1/(1+ np.exp( - self.value)))

    def errorGradient(self, target):
        self.error = (self.value * (1 - self.value) * (target - self.value))

    def calculateNV(self):
        
        for i in range(0, len(self.weights) ):
            for j in range(0, len(self.inputs) ):
                self.value = ((self.weights[i] * self.inputs[j]))

        self.value += self.bias

    def toString(self):
        print("Inputs: ", self.inputs, "Weights: ", self.weights, "Bias: ", self.bias, "Actual Value: ", self.value)

    def totalWeightSummation(self):
        for weight in self.weights:
            self.totalWeights += weight
        
