from imports import *
from models import *

def EXP4(X, Y, verbose=False):

    if verbose:
        print("\n"+40*"#" + " EXP 3: POLY SELECTION ".center(20) + 40*"#")
        print(" We shall consider 3 degrees: 2, 7, 11")
    kfolds = kf(n_splits=3, shuffle=True)

    train_indices = []
    test_indices = []

    for train_index, test_index in kfolds.split(X):
        train_indices.append(train_index)
        test_indices.append(test_index)
    
    xTrain, xTest, yTrain, yTest = indexToSplit(X, Y, train_indices[0], test_indices[0])
    SVMModel(xTrain, xTest, yTrain, yTest, 'poly', verbose, 2)
    xTrain, xTest, yTrain, yTest = indexToSplit(X, Y, train_indices[1], test_indices[1])
    SVMModel(xTrain, xTest, yTrain, yTest, 'poly', verbose, 10)
    xTrain, xTest, yTrain, yTest = indexToSplit(X, Y, train_indices[2], test_indices[2])
    SVMModel(xTrain, xTest, yTrain, yTest, 'poly', verbose, 20)

    if verbose:
        print("\n"+40*"#" + " EXP 3: LINEAR SELECTION ".center(20) + 40*"#")
        print(" We shall consider 3 types of Linear Regressors: Linear, Lasso and Ridge")
    kfolds = kf(n_splits=3, shuffle=True)

    train_indices = []
    test_indices = []

    for train_index, test_index in kfolds.split(X):
        train_indices.append(train_index)
        test_indices.append(test_index)
    
    xTrain, xTest, yTrain, yTest = indexToSplit(X, Y, train_indices[0], test_indices[0])
    LinRegModel(xTrain, xTest, yTrain, yTest, verbose)
    xTrain, xTest, yTrain, yTest = indexToSplit(X, Y, train_indices[1], test_indices[1])
    LassoRegModel(xTrain, xTest, yTrain, yTest, 0.001, 200, verbose)
    xTrain, xTest, yTrain, yTest = indexToSplit(X, Y, train_indices[2], test_indices[2])
    RidgeRegModel(xTrain, xTest, yTrain, yTest, 'auto', 1, verbose)