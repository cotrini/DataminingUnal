from math import sqrt

#this is a class to manage the iris flower data
class IrisFlower:

    def __init__(self, sepalLength, sepalWidth, petalLength, petalWidth, category):
        self.sepalLength = sepalLength
        self.sepalWidth = sepalWidth
        self.petalLength = petalLength
        self.petalWidth = petalWidth
        self.category = category

    # this function returns the distance between two flowers data 
    def euclidenDistanceToFlower(self, flower):
        distance = sqrt(pow(self.sepalLength - flower.sepalLength, 2) + pow(self.sepalWidth - flower.sepalWidth, 2) + pow(self.petalLength - flower.petalLength, 2) + pow(self.petalWidth - flower.petalWidth, 2))
        return distance