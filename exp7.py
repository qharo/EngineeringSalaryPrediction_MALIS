from myLogger import myLog
from imports import *
from models import *

def EXP7(X, Y, verbose):    

    # EXP 7 IS TUNING THE ALPHA VALUES OF RIDGE REGRESSION TO SEE WHETHER OUR MODEL FITS BETTER

    if verbose:
        myLog.heading("EXP 7: RIDGE REGRESSION TUNING")
    kfolds = kf(n_splits=4, shuffle=True)

    train_indices = []
    test_indices = []

    for train_index, test_index in kfolds.split(X):
        train_indices.append(train_index)
        test_indices.append(test_index)

    
    xTrain, xTest, yTrain, yTest = indexToSplit(X, Y, train_indices[0], test_indices[0])

    SOLVERS = ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']

    solverValues = []
    for solver in SOLVERS:
        if verbose: 
            myLog.indent(2, f"For Solver: {solver}")    
        values = []
        highest = 0
        ret = 0
        for alpha in range(0,99, 10):
            if verbose: 
                myLog.indent(2, f'Testing Alpha Value of: {alpha/100}')
            values.append(mae(RidgeRegModel(xTrain, xTest, yTrain, yTest, solver, alpha/100, False).predict(xTest), yTest))
        solverValues += values
    highest = np.argmin(np.array(solverValues))
    myLog.indent(1, f"Lowest Value of MAE: {solverValues[highest]}")
    myLog.indent(1, f"Best Solver: {SOLVERS[highest//10]}")
    myLog.indent(1, f"Best Alpha Value: {(highest%10)/10}")
    



