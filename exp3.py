from myLogger import myLog
from imports import *
from models import *

def EXP3(X, Y, verbose=False):
    if verbose:
        myLog.heading("EXP 3: MODEL SELECTION")
        myLog.indent(1, "4 models to consider: Linear Regression, SVM (Poly), SVM (RBF), SVM (Linear)")
    kfolds = kf(n_splits=4, shuffle=True)

    train_indices = []
    test_indices = []

    for train_index, test_index in kfolds.split(X):
        train_indices.append(train_index)
        test_indices.append(test_index)
    
    xTrain, xTest, yTrain, yTest = indexToSplit(X, Y, train_indices[0], test_indices[0])
    LinRegModel(xTrain, xTest, yTrain, yTest, verbose)
    xTrain, xTest, yTrain, yTest = indexToSplit(X, Y, train_indices[1], test_indices[1])
    if verbose:
        print(" Kernel: Linear")

    print(yTrain.shape)

    SVMModel(xTrain, xTest, yTrain, yTest, 'linear', verbose)
    # xTrain, xTest, yTrain, yTest = indexToSplit(X, Y, train_indices[2], test_indices[2])
    # print(" Kernel: RBF")
    #SVMModel(xTrain, xTest, yTrain, yTest, EXP3, 'rbf')
    # xTrain, xTest, yTrain, yTest = indexToSplit(X, Y, train_indices[3], test_indices[3])
    # print(" Kernel: Poly")
    # SVMModel(xTrain, xTest, yTrain, yTest, EXP3, 'poly')
    if verbose:
        print(" We find that the Linear Regression Model performs well. The SVM also performs well, especially the linear and poly kernel.")