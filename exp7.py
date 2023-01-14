from myLogger import myLog
from imports import *
from models import *

def EXP7(X, Y, verbose):    
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

    SOLVERS = ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']

    solverValues = []
    for solver in SOLVERS:
        if verbose: 
            myLog.indent(2, solver)    
        values = []
        highest = 0
        ret = 0
        for alpha in range(0,100, 10):
            if verbose: 
                myLog.indent(2, f'Test: {alpha/100}')
            values.append(RidgeRegModel(xTrain, xTest, yTrain, yTest, solver, alpha/100, False))
        solverValues += values
    print(np.argmax(np.array(solverValues)))



