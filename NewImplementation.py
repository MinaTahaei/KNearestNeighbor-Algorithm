import pandas as pd
import numpy as np
import math
import operator

data = pd.read_csv('iris.csv')


# euclidean distance
def distance(data1, data2, length):
    eDistance = 0
    for x in range(length):
        eDistance += np.square(data1[x] - data2[x])
    return np.sqrt(eDistance)


# Defining our KNN model
def knn(trainingSet, testInstance, k):
    distances = {}
    sort = {}

    length = testInstance.shape[1]

    for x in range(len(trainingSet)):
        dist = distance(testInstance, trainingSet.iloc[x], length)
        distances[x] = dist[0]

    sorted_d = sorted(distances.items(), key=operator.itemgetter(1))

    neighbors = []

    for x in range(k):
        neighbors.append(sorted_d[x][0])
    classVotes = {}

    for x in range(len(neighbors)):
        response = trainingSet.iloc[neighbors[x]][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1

    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0], neighbors


testSet = [[7.2, 3.6, 5.1, 2.5]]
test = pd.DataFrame(testSet)

print('\n\nWith 1 Nearest Neighbour \n\n')
k = 1
result, neigh = knn(data, test, k)
print('\nPredicted Class of the datapoint = ', result)
print('\nNearest Neighbour of the datapoints = ', neigh)
print('\n\nWith 3 Nearest Neighbours\n\n')
# Setting number of neighbors
k = 3
result, neigh = knn(data, test, k)
print('\nPredicted class of the data = ', result)
print('\nNearest Neighbours of the data = ', neigh)
print('\n\nWith 5 Nearest Neighbours\n\n')
k = 5
result, neigh = knn(data, test, k)
print('\nPredicted class of the data = ', result)
print('\nNearest Neighbours of the data = ', neigh)
