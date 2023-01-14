from myLogger import myLog
from imports import *
from models import *

def EXP6(X, Y, verbose):    
    if verbose:
        myLog.heading("EXP 5: NUMBER OF ESTIMATORS SELECTION")
        #myLog.indent(1, "4 models to consider: Linear Regression, SVM (Poly), SVM (RBF), SVM (Linear)")
    kfolds = kf(n_splits=4, shuffle=True)

    train_indices = []
    test_indices = []

    for train_index, test_index in kfolds.split(X):
        train_indices.append(train_index)
        test_indices.append(test_index)

    
    xTrain, xTest, yTrain, yTest = indexToSplit(X, Y, train_indices[0], test_indices[0])
    if verbose: 
        myLog.indent(2, "1 ESTIMATORS")
    KNNModel(xTrain, xTest, yTrain, yTest, verbose)
    # if verbose: 
    #     myLog.indent(2, "5 ESTIMATORS")
    # RFCModel(xTrain, xTest, yTrain, yTest, 5, verbose)
    # if verbose: 
    #     myLog.indent(2, "10 ESTIMATORS")
    # RFCModel(xTrain, xTest, yTrain, yTest, 10, verbose)
    # if verbose: 
    #     myLog.indent(2, "100 ESTIMATORS")
    # RFCModel(xTrain, xTest, yTrain, yTest, 100, verbose)