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

    MAE_SolverValues = []
    MSE_SolverValues = []
    for solver in SOLVERS:
        if verbose: 
            myLog.indent(2, f"For Solver: {solver}")    
        mae_values = []
        mse_values = []
        highest = 0
        ret = 0
        for alpha in range(0,99, 10):
            if verbose: 
                myLog.indent(2, f'Testing Alpha Value of: {alpha/100}')
            mae_values.append(mae(RidgeRegModel(xTrain, xTest, yTrain, yTest, solver, alpha/100, False).predict(xTest), yTest))
            mse_values.append(mse(RidgeRegModel(xTrain, xTest, yTrain, yTest, solver, alpha/100, False).predict(xTest), yTest))
        MAE_SolverValues += mae_values
        MSE_SolverValues += mse_values
    highest = np.argmin(np.array(MAE_SolverValues))
    myLog.indent(1, f"Lowest Value of MAE: {MAE_SolverValues[highest]}")
    myLog.indent(1, f"Lowest Value of MSE: {MSE_SolverValues[highest]}")
    myLog.indent(1, f"Lowest Value of MSE: {np.sqrt(MSE_SolverValues[highest])}")
    best_solver = SOLVERS[highest//10]
    best_alpha = (highest%10)/10
    myLog.indent(1, f"Best Solver: {best_solver}")
    myLog.indent(1, f"Best Alpha Value: {best_alpha}")

    xTrain, xTest, yTrain, yTest = tts(X, Y, test_size=0.1)    
    myLog.heading("Best Ridge Regression Model")
    RidgeRegModel(xTrain, xTest, yTrain, yTest, best_solver, best_alpha, True)




