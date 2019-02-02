from mnist import MNIST
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.image as mpimg
from tqdm import tqdm
from skimage.transform import resize
from sklearn.ensemble import RandomForestClassifier
from itertools import product
from scipy.stats import norm

mndata = MNIST('./data/mnist/')

def showImage(pixels, label, shape = (28, 28)):
    pixels = np.array(pixels, dtype='uint8')
    pixels = pixels.reshape(shape)
    plt.title('Label is {label}'.format(label=label))
    plt.imshow(pixels, cmap='gray')
    plt.show()

def aggregateImagesByNumber(images, labels): 
    numberToImages = {}
    for index in range(len(images)):
        label = labels[index]
        if(label not in numberToImages):
            numberToImages[label] = []
        numberToImages[label].append(images[index])
    return numberToImages

def calculateMeanAndVar(data):
    results = []
    for item in zip(*data):
        meanVar = (np.nanmean(item), np.nanvar(item))
        results.append(meanVar)
    return results

def calculateMeanAndVarForNumbers(dataByClasses):
    results = {}
    for classValue, values in dataByClasses.items():
        results[classValue] = calculateMeanAndVar(values)
    return results

def calculateProbability(x, mean, var):
    p = 1/(np.sqrt(2*np.pi*var)) * np.exp((-(x-mean)**2)/(2*var))
    return p

def calculateProbabilitiesForNumbers(dataByClasses, imagePixels):
    classProbabilities = {}
    for label, images in dataByClasses.items():
        classProbabilities[label] = 1
        for i in range(len(images)):
            mean, var = images[i]
            inputV = imagePixels[i]
            prob = calculateProbability(inputV, mean, var)
            classProbabilities[label] *= prob
    return classProbabilities

def getPrediction(numberSummaries, imagePixels):
    numberProbabilities = calculateProbabilitiesForNumbers(numberSummaries, imagePixels)
    predictedLabel, bestProb = None, -1
    for label, prob in numberProbabilities.items():
        if predictedLabel is None or prob > bestProb:
            predictedLabel = label
            bestProb = prob
    return predictedLabel

def getPredictionsForTestNumbers(numberSummaries, testImages):
    predictions = []
    for imagePixels in tqdm(testImages):
        prediction = getPrediction(numberSummaries, imagePixels)
        predictions.append(prediction)
    return predictions

def getAccuracy(predictions, testData, testLabels):
    accurate = 0
    rows_in_test_set = len(testData)
    for index in range(rows_in_test_set):
        if predictions[index] == testLabels[index]:
            accurate += 1
    return (accurate / rows_in_test_set) * 100

def thresholdOriginalImages(images):
    threshold, upper, lower = 127, 1, 0
    return np.where(np.array(images)>threshold, upper, lower)

def thresholdScaledImages(imagePixels):
    a = imagePixels.copy()
    grey = a[a > 0]
    mid = np.sum(grey) / len(grey)
    return np.where(np.array(a)>mid, 1, 0)

def resizeImage(imagePixels, size):
    xmax, ymax = np.max(np.where(imagePixels!=0), 1)
    xmin, ymin = np.min(np.where(imagePixels!=0), 1)
    bounding = imagePixels[xmin:xmax,ymin:ymax]
    resized = resize(bounding, size, mode='constant')
    return resized

def train(trainingData, trainingLabels, testData, testLabels):
    print("Grouping by class")
    groupedByClass = aggregateImagesByNumber(trainingData, trainingLabels)
    print("Calculating mean and var")
    classSummary = calculateMeanAndVarForNumbers(groupedByClass)
    
    print("Getting Predictions for test images")
    predictions = getPredictionsForTestNumbers(classSummary, testData)
    print("Getting Accuracy for test images")
    test_accuracy =  getAccuracy(predictions, testData, testLabels)
    print(f"Accuracy for test images: {test_accuracy}")
    
    print("Getting Predictions for training images")
    predictions = getPredictionsForTestNumbers(classSummary, trainingData)
    print("Getting Accuracy for training images")
    training_accuracy =  getAccuracy(predictions, trainingData, trainingLabels)
    print(f"Accuracy for training images: {training_accuracy}")
    return test_accuracy, training_accuracy

trainingDataImages, trainingLabels = mndata.load_training()
trainingDataImages = np.array(trainingDataImages).astype(np.uint8)
testDataImages, testLabels = mndata.load_testing()
testDataImages = np.array(testDataImages).astype(np.uint8)
print(f"Loaded {len(trainingDataImages)} training images and {len(testDataImages)} test images")

trainingImagesCleaned = thresholdOriginalImages(trainingDataImages)
testImagesCleaned = thresholdOriginalImages(testDataImages)
print(f"Applied threshold to training and test images")
showImage(trainingDataImages[2], trainingLabels[2])
showImage(trainingImagesCleaned[2], trainingLabels[2])

size = (20, 20)
print(f"Resizing images to {size}")
resizedTrainingImages = np.array([np.ravel(thresholdScaledImages(resizeImage(x, size))) for x in trainingDataImages.reshape((-1, 28, 28))])
resizedTestImages = np.array([np.ravel(thresholdScaledImages(resizeImage(x, size))) for x in testDataImages.reshape((-1, 28, 28))])
showImage(resizedTrainingImages[2], trainingLabels[2], size)

train(trainingImagesCleaned, trainingLabels, testImagesCleaned, testLabels)

train(resizedTrainingImages, trainingLabels, resizedTestImages, testLabels)

def countByLabel(labels):
    unique, count = np.unique(labels, return_counts=True)
    return dict(zip(unique, count))

def calculatePdfLog(x, mean):
        if x == 0:
            result = 1 - mean
        else:
            result = mean
        if result == 0.0:
            return 0
        return math.log(result, 10)

def getPredictionForImages(images, means, priors, counts_by_label): 
    predictions = []
    for image in tqdm(images):
        results = []
        for label, counts in counts_by_label.items():
            prior_log = np.log(priors[label])
            posterior_log = np.sum([calculatePdfLog(image[feature], means[label][feature]) for feature in range(len(means[label]))])
            likelyhood = prior_log + posterior_log
            results.append(likelyhood)    
        predictions.append(np.argmax(results))
    return predictions

def calculateBernoulli(images, labels, counts_by_label):
    training_size = len(images)
    label_means = {}
    
    for label, counts in counts_by_label.items():
        sum = np.sum(images[i] if labels[i] == label else 0.0 for i in range(training_size))    
        label_means[label] = sum / counts 
        
    label_prior = {label:(counts/training_size) for label, counts in list(counts_by_label.items())}
    
    return label_means, label_prior

def trainWithBernoulli(trainingImages, trainingLabels, testImages, testLabels):   
    counts_by_label = countByLabel(trainingLabels)
    print("Calculating Bernoulli") 
    label_means, label_prior = calculateBernoulli(trainingImages, trainingLabels, counts_by_label)
    
    print("Getting Predictions for test images")
    predictions = getPredictionForImages(testImages, label_means, label_prior, counts_by_label)
    print("Getting Accuracy for test images")
    test_accuracy =  getAccuracy(predictions, testImages, testLabels)
    print(f"Accuracy for test images: {test_accuracy}")
    
    print("Getting Predictions for training images")
    predictions = getPredictionForImages(trainingImages, label_means, label_prior, counts_by_label)
    print("Getting Accuracy for training images")
    training_accuracy =  getAccuracy(predictions, trainingImages, trainingLabels)
    print(f"Accuracy for training images: {training_accuracy}")
    return accuracy

trainWithBernoulli(trainingImagesCleaned, trainingLabels, testImagesCleaned, testLabels)
trainWithBernoulli(resizedTrainingImages, trainingLabels, resizedTestImages, testLabels)

def trainWithRandomForest(trainImages, trainLabels, testImages, testLabels):
    num_of_trees = [10, 30]
    num_of_depths = [4, 16]
    for [tree, depth] in list(product(num_of_trees, num_of_depths)): 
        classifier = RandomForestClassifier(n_estimators=tree, max_depth=depth)
        classifier.fit(trainImages, trainLabels)
        predictions = classifier.predict(testImages)
        test_accuracy =  getAccuracy(predictions, testImages, testLabels)
        print(f"Accuracy for test images: {test_accuracy} using n_estimators={tree} and max_depth={depth}")
        
        predictions = classifier.predict(trainImages)
        training_accuracy =  getAccuracy(predictions, trainImages, trainLabels)
        print(f"Accuracy for training images: {training_accuracy} using n_estimators={tree} and max_depth={depth}")

trainWithRandomForest(trainingImagesCleaned, trainingLabels, testImagesCleaned, testLabels)
trainWithRandomForest(resizedTrainingImages, trainingLabels, resizedTestImages, testLabels)

