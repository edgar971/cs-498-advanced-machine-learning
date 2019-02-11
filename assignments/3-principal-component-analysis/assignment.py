get_ipython().magic('matplotlib inline')
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot

def loadData():
    column_names = ['sepal_length','sepal_width','petal_length','petal_width']
    data1 = np.array(pd.read_csv('./data/dataI.csv', names=column_names, skiprows=1))
    data2 = np.array(pd.read_csv('./data/dataII.csv', names=column_names, skiprows=1))
    data3 = np.array(pd.read_csv('./data/dataIII.csv', names=column_names, skiprows=1))
    data4 = np.array(pd.read_csv('./data/dataIV.csv', names=column_names, skiprows=1))
    data5 = np.array(pd.read_csv('./data/dataV.csv', names=column_names, skiprows=1))
    data = np.array(pd.read_csv('./data/iris.csv', names=column_names, skiprows=1))
    return [data, data1, data2, data3, data4, data5]

def createPCA(data, n_components):
    pca = PCA(n_components=n_components)
    pca.fit(data)
    return pca

def calculateMeanSquareMean(dataA, dataB):
    total = 0
    number_of_features = 4
    for i in range(len(dataA)):
        for feature in range(number_of_features):
            total += np.square(dataA[i, feature] - dataB[i, feature])
    return total / len(dataA)

def generateSubmission(dataA, dataB, name):
    headers = '0N,1N,2N,3N,4N,0c,1c,2c,3c,4c'
    results = np.concatenate((dataA, dataB), axis=1)
    np.savetxt(f"{name}-numbers.csv", results, header=headers, delimiter=",", fmt='%f')

datasets = loadData()

def train(noisy_datasets, noiseless, max_components, fit_on_noisy=False):
    results = []
    
    for i in range(len(noisy_datasets)):
        errors = []
        for n_components in range(max_components):
            fit_data = noisy_datasets[i] if fit_on_noisy else noiseless
            noiseless_pca = createPCA(fit_data, n_components)
            transformed = noiseless_pca.transform(noisy_datasets[i])
            inverted = noiseless_pca.inverse_transform(transformed)
            errors.append(calculateMeanSquareMean(inverted, noiseless))
        results.append(errors)
    return results

def createReconstruction(data, n_components=2):
    pca = createPCA(data, n_components)
    transformed = pca.transform(data)
    reconstructed = pca.inverse_transform(transformed)
    return reconstructed

noiseless_msn = train(datasets[1:], datasets[0], 5)
noisy_msn = train(datasets[1:], datasets[0], 5, True)

generateSubmission(noisy_msn, noiseless_msn, "edgarsp2")

reconstructed = createReconstruction(datasets[1])
headers = 'Sepal.Length,Sepal.Width,Petal.Length,Petal.Width'

np.savetxt("edgarsp2-recon.csv", reconstructed, header=headers, delimiter=",", fmt='%f')

