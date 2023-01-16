from myLogger import myLog
from imports import *
from models import *

def EXP6(X, Y, verbose):    
    if verbose:
        myLog.heading("EXP 6: KNN")
        #myLog.indent(1, "4 models to consider: Linear Regression, SVM (Poly), SVM (RBF), SVM (Linear)")
    kfolds = kf(n_splits=4, shuffle=True)

    train_indices = []
    test_indices = []

    for train_index, test_index in kfolds.split(X):
        train_indices.append(train_index)
        test_indices.append(test_index)

    TEST_VALUES = [1, 5, 10, 100]

    for i in range(0, 4):
        xTrain, xTest, yTrain, yTest = indexToSplit(X, Y, train_indices[i], test_indices[i])
        if verbose: 
            myLog.indent(2, f"{TEST_VALUES[i]} ESTIMATORS")
        KNNModel(xTrain, xTest, yTrain, yTest, TEST_VALUES[i], verbose)

    if verbose:
        myLog.indent(1, "We find that 10 neighbours works best, so experiment again with finer values")

    TEST_VALUES = [10, 30, 60, 90]

    for i in range(0, 4):
        xTrain, xTest, yTrain, yTest = indexToSplit(X, Y, train_indices[i], test_indices[i])
        if verbose: 
            myLog.indent(2, f"{TEST_VALUES[i]} ESTIMATORS")
        KNNModel(xTrain, xTest, yTrain, yTest, TEST_VALUES[i], verbose)

    
    if verbose:
        myLog.indent(1, "We find that beween 5-10 neighbours works best, so experiment again with finer values")

    TEST_VALUES = [5, 7, 9, 11]

    for i in range(0, 4):
        xTrain, xTest, yTrain, yTest = indexToSplit(X, Y, train_indices[i], test_indices[i])
        if verbose: 
            myLog.indent(2, f"{TEST_VALUES[i]} ESTIMATORS")
        KNNModel(xTrain, xTest, yTrain, yTest, TEST_VALUES[i], verbose)
    
    myLog.indent(1, "7 - 9 seems to yield best results after multiple tests")
