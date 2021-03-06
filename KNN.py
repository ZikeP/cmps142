#!/usr/bin/env python

import sys
import getopt
import numpy as np
import math
import operator

def loadCSV(str):
    tmp = np.loadtxt(str, dtype=np.str, skiprows=1, delimiter=",")
    data = tmp.astype(np.int)
    return data;

def l1distance(instance1, instance2):
    distance = 0;
    for x in range(4):
        distance+=abs(instance1[x]-instance2[x])
    return distance

def l2distance(instance1, instance2):
    distance = 0;
    for x in range(4):
        distance+=pow((instance1[x]-instance2[x]),2)
    return math.sqrt(distance)

def linfdistance(instance1, instance2):
    distance = 0;
    for x in range(4):
        max=abs(instance1[x]-instance2[x])
        if distance < max:
            distance = max

    return distance

#search k of the neariest neighbors of testinstance by using method l1,l2 or linf
def getNeighbors(trainset, testinstance, k, method):
    distances = []
    if method == "L1":
        for x in range(len(trainset)):
            dist = l1distance(testinstance, trainset[x])
            distances.append((trainset[x], dist))
        distances.sort(key=operator.itemgetter(1))
        neighbors = []
        for x in range(k):
            neighbors.append(distances[x][0])
        return neighbors
    elif method == "L2":
        for x in range(len(trainset)):
            dist = l2distance(testinstance, trainset[x])
            distances.append((trainset[x], dist))
        distances.sort(key=operator.itemgetter(1))
        neighbors = []
        for x in range(k):
            neighbors.append(distances[x][0])
        return neighbors
    elif method == "Linf":
        for x in range(len(trainset)):
            dist = linfdistance(testinstance, trainset[x])
            distances.append((trainset[x], dist))
        distances.sort(key=operator.itemgetter(1))
        neighbors = []
        for x in range(k):
            neighbors.append(distances[x][0])
        return neighbors

#collect the label of neighbors, and calculate the sum, if sum >0, means most of label is 1 and vice versa.
def getResponse(neighbors):
    label = 0
    for x in range(len(neighbors)):
        label += neighbors[x][-1]
    if label >= 0:
        return 1
    if label <0:
        return -1

def getAccuracy(testset, predictions):
    correct = 0
    for x in range(len(testset)):
        if testset[x][-1] == predictions[x]:
            correct+=1
    return (correct/float(len(testset))) * 100.0

k = 0
method = ""
try:
    try:
        opts, args = getopt.getopt(sys.argv[1:],"-K:-method:", ["K=","method="])
    except getopt.GetoptError:
        print("Usage: python KNN.py --K <the value of K> --method <L1, L2, or Linf>")
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-k","--K"):
            k = int(arg)
        elif opt in ("-m", "--method"):
            method = arg

    testset = loadCSV("knn_test.csv")
    trainset = loadCSV("knn_train.csv")
    predictions = []
    # k = 3
    # method = "L2"
    for x in range(len(testset)):
        neighbors = getNeighbors(trainset, testset[x], k, method)
        result = getResponse(neighbors)
        predictions.append(result)
        print('predicted=' + repr(result) + ', actual=' + repr(testset[x][-1]))
    accuracy = getAccuracy(testset, predictions)
    print('Accuracy: ' + repr(accuracy) + '%')
except TypeError:
    print("Usage: python KNN.py --K <the value of K> --method <L1, L2, or Linf>")
    exit()
