import pandas as pd
import numpy as np
import random
import math
from functools import reduce
from numpy import genfromtxt
import copy 

def getData():
    data = np.genfromtxt('./data/pima-indians-diabetes.csv', delimiter=',')
    return data[1:,:]

def split(dataset, training_percent = .8):
    np.random.shuffle(dataset)
    num_of_items = len(dataset)
    training_split = int(.8 * num_of_items)
    return dataset[:training_split], dataset[training_split:]

def aggregateByClass(data): 
    classToValues = {}
    for item in data:
        classType = item[-1]
        if(classType not in classToValues):
            classToValues[classType] = []
        classToValues[classType].append(item)
    return classToValues

def calculateMeanAndVar(data):
    results = []
    for item in zip(*data):
        results.append((np.nanmean(item), np.nanvar(item)))
    del results[-1]
    return results

def calculateForClass(dataByClasses):
    results = {}
    for classValue, values in dataByClasses.items():
        results[classValue] = calculateMeanAndVar(values)
    return results

def calculateProbability(x, mean, var):
    p = 1/(np.sqrt(2*np.pi*var)) * np.exp((-(x-mean)**2)/(2*var))
    return p

def calculateProbabilitiesForClass(dataByClasses, vector):
    classProbabilities = {}
    for classType, classSummaries in dataByClasses.items():
        classProbabilities[classType] = 1
        for i in range(len(classSummaries)):
            mean, var = classSummaries[i]
            inputV = vector[i]
            classProbabilities[classType] *= calculateProbability(inputV, mean, var)
    return classProbabilities

def makePrediction(classSummaries, vector):
    classProbabilities = calculateProbabilitiesForClass(classSummaries, vector)
    predictedLabel, bestProb = None, -1
    for label, prob in classProbabilities.items():
        if predictedLabel is None or prob > bestProb:
            predictedLabel = label
            bestProb = prob
    return predictedLabel

def getPredictionsForClass(classSummaries, testDataset):
    predictions = []
    for testValue in testDataset:
        prediction = makePrediction(classSummaries, testValue)
        predictions.append(prediction)
    return predictions

def getAccuracy(predictions, testData):
    accurate = 0
    rows_in_test_set = len(testData)
    for index in range(rows_in_test_set):
        if predictions[index] == testData[index][-1]:
            accurate += 1
    return (accurate / rows_in_test_set) * 100

def train(data):
    splits = 10
    totalAccuracy = 0
    for step in range(splits):
        trainingData, testData = split(data)
        groupedByClass = aggregateByClass(trainingData)
        classSummary = calculateForClass(groupedByClass)
        predictions = getPredictionsForClass(classSummary, testData)
        accuracy = getAccuracy(predictions, testData)
        totalAccuracy += accuracy
    return totalAccuracy / splits

data = getData()
train(data)

def dropZeros(data):
    columns = [2,3,5,7]
    newData = []
    for i in range(len(data)):
        newArray = []
        for x in range(len(data[i])):
            if x in columns and data[i][x] == 0:
                continue
            else:
                newArray.append(data[i][x])
        newData.append(newArray)
    return np.array(newData)

data = getData()
cleanedData = dropZeros(data)
train(cleanedData)

