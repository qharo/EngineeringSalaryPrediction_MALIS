from myLogger import myLog
from imports import *
from models import *

def EXP8(X, Y, verbose):    
    if verbose:
        myLog.heading("EXP 7: RIDGE REGRESSION TUNING")
        #myLog.indent(1, "4 models to consider: Linear Regression, SVM (Poly), SVM (RBF), SVM (Linear)")
    kfolds = kf(n_splits=4, shuffle=True)

    train_indices = []
    test_indices = []

    for train_index, test_index in kfolds.split(X):
        train_indices.append(train_index)
        test_indices.append(test_index)

    
    xTrain, xTest, yTrain, yTest = indexToSplit(X, Y, train_indices[0], test_indices[0])

    values = []
    for iter in range(1,1_000, 100):
        if verbose: 
            myLog.indent(2, f'Iterations: {iter}')
        values.append(LassoRegModel(xTrain, xTest, yTrain, yTest, 100, True))
    print(np.argmax(values))