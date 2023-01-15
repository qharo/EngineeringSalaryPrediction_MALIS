from myLogger import myLog
from imports import *
from models import *

def EXP5(X, Y, verbose):    
    if verbose:
        myLog.heading("EXP 5: NUMBER OF ESTIMATORS SELECTION")
        myLog.indent(1, "We choose 4 values of estimators: 1, 5, 10, 100")
    kfolds = kf(n_splits=4, shuffle=True)

    train_indices = []
    test_indices = []

    for train_index, test_index in kfolds.split(X):
        train_indices.append(train_index)
        test_indices.append(test_index)

    
    xTrain, xTest, yTrain, yTest = indexToSplit(X, Y, train_indices[0], test_indices[0])

    TEST_VALUES = [100, 500, 250, 750]

    for i in range(0, 4):
        xTrain, xTest, yTrain, yTest = indexToSplit(X, Y, train_indices[i], test_indices[i])
        if verbose: 
            myLog.indent(2, f"{TEST_VALUES[i]} TREES")
        RFCModel(xTrain, xTest, yTrain, yTest, TEST_VALUES[i], verbose)

    