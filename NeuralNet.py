
import random
from tkinter import N
from Neuron import Neuron
import numpy as np


class NeuralNet:

    def __init__(self, amountInputLayers, amountHiddenLayers, amountOutputLayers):
        self.errors = []
        # self.inputLayers = [Neuron() for each in range(amountInputLayers)]
        self.hiddenLayers = [Neuron() for each in range(amountHiddenLayers)]
        self.outputLayers = [Neuron() for each in range(amountOutputLayers)]
        # for i in range(0, amountInputLayers):
        #     self.inputLayers[i].weights = [random.uniform(-1, 1) for each in range(amountHiddenLayers)]
        #     self.inputLayers[i].bias = random.uniform(-1, 1)
        for i in range(0, amountHiddenLayers):
            self.hiddenLayers[i].weights = [(random.uniform(-1, 1)) for each in range(amountInputLayers)]
            self.hiddenLayers[i].bias = (random.uniform(-1, 1))
        for i in range(0, amountOutputLayers):
            self.outputLayers[i].weights = [(random.uniform(-1, 1)) for each in range(amountHiddenLayers)]
            self.outputLayers[i].bias = (random.uniform(-1, 1))
        
    def calculeNetValue(self, inputValues):
        outputs = []
        # for i in range(0, len(self.inputLayers)):
        #     self.inputLayers[i].inputs = [inputValues[i],]
        #     self.inputLayers[i].calculateNV()
        # for i in range(0, len(self.hiddenLayers)):
        #     self.hiddenLayers[i].inputs = [self.inputLayers[each].value for each in range(0, len(self.inputLayers))]
        #     self.hiddenLayers[i].calculateNV()
        #     output = (1/(1+ np.exp( - self.hiddenLayers[i].value)))
        #     outputs.append(output)

        #this part is to calculate the net value from layers
        for hidenNeuron in self.hiddenLayers:
            hidenNeuron.inputs = [inputValues[each] for each in range(0, len(inputValues))]
            hidenNeuron.calculateNV()

        for outputNeuron in self.outputLayers:
            outputNeuron.inputs = [self.hiddenLayers[each].value for each in range(0, len(self.hiddenLayers))]
            outputNeuron.calculateNV()

        #this part is to calculate the output with activation function from layers

        for outputNeuron in self.outputLayers:
            outputNeuron.activationFunction()
            output = outputNeuron.output
            outputs.append(output)
            
        for hidenNeuron in self.hiddenLayers:
            hidenNeuron.activationFunction()
            output = hidenNeuron.output
            outputs.append(output)

        # for i in range(0, len(self.outputLayers)):
        #     self.outputLayers[i].inputs = [self.hiddenLayers[each].value for each in range(0, len(self.hiddenLayers))]
        #     self.outputLayers[i].calculateNV()
        #     output = (1/(1+ np.exp( - self.outputLayers[i].value)))
        #     outputs.append(output)

        return outputs


    def train(self, inputValues, targetValues):
        outputs = self.calculeNetValue(inputValues)
        errors = []

        # this section is to calculate error from output layers
        for i in range(0, len(self.outputLayers)):
            self.outputLayers[i].errorGradient(targetValues[i])
            errors.append(self.outputLayers[i].error)
            for j in range(0, len(self.hiddenLayers)):
                self.hiddenLayers[j].totalWeightSummation()
                self.hiddenLayers[j].error = ((self.hiddenLayers[j].output * (1 - self.hiddenLayers[j].output) * (self.outputLayers[i].error * self.hiddenLayers[j].totalWeights)))
        self.errors = errors     
        # this section is to calculate error from hiden layers
        # for i in range(0, len(self.hiddenLayers)):
        #     error= (outputs[i+ len(self.outputLayers)] * (1 - outputs[i + len(self.outputLayers)])) * sum(errors[each] * self.hiddenLayers[i].weights[each] for each in range(0, len(self.outputLayers)))
        #     errors.append(error)
        # self.errors = errors

        # this section is to update weights and bias from output layers
        for error in errors:
            for i in range(0, len(self.hiddenLayers)):
                for j in range(0, len(self.hiddenLayers[i].weights)):   
                    self.hiddenLayers[i].weights[j] += (error * self.hiddenLayers[i].output * 0.9)
                self.hiddenLayers[i].bias += (error * 0.9)
    


            for o in range(0, len(self.outputLayers)):
                for k in range(0, len(self.outputLayers[o].weights)):
                    self.outputLayers[o].weights[k] += ((error * self.outputLayers[o].output * 0.9))
                self.outputLayers[o].bias += (error * 0.9)

        

        # for i in range(0, len(self.outputLayers)):
        #     if errors[i] != 0:
        #         for j in range(0, len(self.hiddenLayers)):
        #             self.outputLayers[i].weights[j] += outputs[i] * errors[i] * 0.9
        #         self.outputLayers[i].bias += (errors[i] * 0.9)

       
        # for i in range(0, len(self.hiddenLayers)):
        #     if errors[i+len(self.outputLayers)] != 0:
        #         for j in range(len(self.inputLayers)):
        #             self.hiddenLayers[i].weights[j] += (outputs[i + len(self.outputLayers)] * errors[i] * 0.9)
        #         self.hiddenLayers[i].bias += (errors[i+len(self.outputLayers)] * 0.9)
        

    def toString(self, value):
        # print("_________________________________________________")
        # print("Input Layers: ", "Cicle: ",  value)
        # print("_________________________________________________")
        # for i in range(0, len(self.inputLayers)):
        #     print("Input Layer: ", i)
        #     self.inputLayers[i].toString()
        print("_________________________________________________")
        print("Hidden Layers: ", "Cicle: ",  value)
        print("_________________________________________________")
        for i in range(0, len(self.hiddenLayers)):
            print("Hidden Layer: ", i)
            self.hiddenLayers[i].toString()
        print("_________________________________________________")
        print("Output Layers: ", "Cicle: ",  value)
        print("_________________________________________________")
        for i in range(0, len(self.outputLayers)):
            print("Output Layer: ", i)
            self.outputLayers[i].toString()
        print("_________________________________________________")  
        print("Errors: ", "Cicle: ",  value)
        print("_________________________________________________")
        print(self.errors,"\n")