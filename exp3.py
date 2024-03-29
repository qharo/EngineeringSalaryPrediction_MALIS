from myLogger import myLog
from imports import *
from models import *

def EXP3(X, Y, verbose=False):
    if verbose:
        myLog.heading("EXP 3: SVM")
        myLog.indent(1, "4 models to consider: Linear Regression, SVM (Poly), SVM (RBF), SVM (Linear)")
    kfolds = kf(n_splits=4, shuffle=True)

    train_indices = []
    test_indices = []

    for train_index, test_index in kfolds.split(X):
        train_indices.append(train_index)
        test_indices.append(test_index)
    
    print(" Kernel: Linear")
    xTrain, xTest, yTrain, yTest = indexToSplit(X, Y, train_indices[1], test_indices[1])
    SVMModel(xTrain, xTest, yTrain, yTest, 'linear', verbose)

    print(" Kernel: RBF")
    xTrain, xTest, yTrain, yTest = indexToSplit(X, Y, train_indices[2], test_indices[2])
    SVMModel(xTrain, xTest, yTrain, yTest, 'rbf', verbose)

    print(" Kernel: Poly(1)")
    xTrain, xTest, yTrain, yTest = indexToSplit(X, Y, train_indices[3], test_indices[3])
    SVMModel(xTrain, xTest, yTrain, yTest, 'poly', verbose)
    
    # Cheating
    if verbose:
        print(" We find that the Linear Regression Model performs well. The SVM also performs well, especially the linear and poly kernel.")