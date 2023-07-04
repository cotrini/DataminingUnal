# import pandas as pd
import numpy as np
from NeuralNet import NeuralNet

#this part of code is for replace the Variety to numbers only
# df = pd.read_csv('iris.csv').replace('Setosa', 0).replace('Versicolor', 1).replace('Virginica', 2)
# df.to_csv('irisNumbersOnly.csv', index=False)

# this part of code is for normalice the data with z-score
irisData = np.genfromtxt('irisNumbersOnly.csv', delimiter=',')

def zscoreNormalizer(value, mean, std):
    return np.around(((value - mean) / std),5)


for i in range(0, irisData.shape[0]):
    for j in range(0, irisData.shape[1]-1):
        irisData[i][j] =  zscoreNormalizer(irisData[i,j], np.mean(irisData[:,j]), np.std(irisData[:,j]))



#this code from de book is to make vector of variety of flowers in the same order of de data to compare the result
target = np.zeros((np.shape(irisData)[0],3))
indices = np.where(irisData[:,4]==0)
target[indices,0] = 1
indices = np.where(irisData[:,4]==1)
target[indices,1] = 1
indices = np.where(irisData[:,4]==2)
target[indices,2] = 1

#un order data
order = []
for o in range(0, irisData.shape[0]):
    order.append(o)

np.random.shuffle(order)
irisData = irisData[order,:]
target = target[order,:]

myNeuralNet = NeuralNet(4,5,3)
#hold on part

trainingDataSize = int(input("Enter the size of training data default 105 flowers(75%): ") or 105)
trainingData = irisData[0:trainingDataSize,:]
trainingTarget = target[0:trainingDataSize,:]
testData = irisData[trainingDataSize:,:]
testTarget = target[trainingDataSize:,:]
acuarcy = 0
fails = 0

#this part of code is for insert data training


for j in range(0, 1):
    myNeuralNet.toString(j)
    for i in range(0, trainingDataSize):
        myNeuralNet.train(trainingData[i,:], trainingTarget[i,:])



#this part of code is for test the data

for i in range(0, testData.shape[0]):
 
    auxOutput = myNeuralNet.calculeNetValue(testData[i,:])
    output = auxOutput[:3]
    #myNeuralNet.toString(i)
    if np.argmax(output) == np.argmax(testTarget[i,:]):
        acuarcy += 1
        print('Output: ', output, 'Target: ', testTarget[i,:], 'Hit')
    else:
        fails += 1
        print('Output: ', output, 'Target: ', testTarget[i,:], 'Fail')
    

print('amount of hits: ',acuarcy)
print('amount of fails: ',fails)
print('percent of acuarcy: ',(acuarcy/testData.shape[0])*100,'%')


sepalLength = float(input("Enter the sepal length: ") or 5.1)
sepalLength = zscoreNormalizer(sepalLength, np.mean(irisData[:,0]), np.std(irisData[:,0]))
sepalWidth = float(input("Enter the sepal width: ") or 3.5)
sepalWidth = zscoreNormalizer(sepalWidth, np.mean(irisData[:,1]), np.std(irisData[:,1]))
petalLength = float(input("Enter the petal length: ") or 1.4)
petalLength = zscoreNormalizer(petalLength, np.mean(irisData[:,2]), np.std(irisData[:,2]))
petalWidth = float(input("Enter the petal width: ") or 0.2)
petalWidth = zscoreNormalizer(petalWidth, np.mean(irisData[:,3]), np.std(irisData[:,3]))

flowerVector = myNeuralNet.calculeNetValue([sepalLength,sepalWidth,petalLength,petalWidth])

print('The flower is: ',flowerVector[0:3] )

