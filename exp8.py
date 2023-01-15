from myLogger import myLog
from imports import *
from models import *

def EXP8(X, Y, verbose):    

    # EXP 7 IS TUNING THE NUMBER OF ITERATIONS OF RIDGE REGRESSION TO SEE WHETHER OUR MODEL FITS BETTER

    if verbose:
        myLog.heading("EXP 8: LASSO REGRESSION TUNING")
    kfolds = kf(n_splits=4, shuffle=True)

    train_indices = []
    test_indices = []

    for train_index, test_index in kfolds.split(X):
        train_indices.append(train_index)
        test_indices.append(test_index)

    
    #xTrain, xTest, yTrain, yTest = indexToSplit(X, Y, train_indices[0], test_indices[0])
    xTrain, xTest, yTrain, yTest = tts(X, Y, test_size=0.1)

    tol_values = []
    values = []
    for tol in [1e-4, 1e-3, 1e-2, 1, 10, 100]:
        for iter in range(1,1_000, 100):
            if verbose: 
                myLog.indent(2, f'Iterations: {iter}')
            values.append(mae(LassoRegModel(xTrain, xTest, yTrain, yTest, tol, iter, False).predict(xTest), yTest))
        myLog.indent(1, f"Best value of number of iterations: {range(1, 1_1000, 100)[np.argmin(values)]} with a MAE of {min(values)}")
        tol_values.append(min(values))
    myLog.indent(1, f"Best value of tol is {[1e-4, 1e-3, 1e-2, 1][np.argmin(tol_values)]}")


    tol_values = []
    values = []
    for tol in [1e-4, 5e-4, 1e-5]:
        for iter in range(1,200, 10):
            if verbose: 
                myLog.indent(2, f'Iterations: {iter}')
            values.append(mae(LassoRegModel(xTrain, xTest, yTrain, yTest, tol, iter, False).predict(xTest), yTest))
        myLog.indent(1, f"Best value of number of iterations: {range(1, 200, 10)[np.argmin(values)]} with a MAE of {min(values)}")
        tol_values.append(min(values))
    myLog.indent(1, f"Best value of tol is {[1e-4, 5e-4, 1e-5][np.argmin(tol_values)]}")