from myLogger import myLog
from imports import *
from models import *

def EXP8(X, Y, verbose):    

    salary_sd = 70429.13957426547 

    # EXP 8 IS TUNING THE NUMBER OF ITERATIONS OF LASSO REGRESSION TO SEE WHETHER OUR MODEL FITS BETTER

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

    TOL = [1e-4, 1e-3, 1e-2, 1, 10, 100]

    MAE_SolverValues = []
    MSE_SolverValues = []
    for tol in TOL:
        if verbose: 
            myLog.indent(2, f"For tol: {tol}")    
        mae_values = []
        mse_values = []
        for iter in range(1,1_000, 100):
            if verbose: 
                myLog.indent(2, f'Testing Iter of: {iter}')
            mae_values.append(mae(LassoRegModel(xTrain, xTest, yTrain, yTest, tol, iter, True).predict(xTest), yTest))
            mse_values.append(mse(LassoRegModel(xTrain, xTest, yTrain, yTest, tol, iter, True).predict(xTest), yTest))
        MAE_SolverValues += mae_values
        MSE_SolverValues += mse_values
    highest = np.argmin(np.array(MAE_SolverValues))
    myLog.indent(1, f"Lowest Value of MAE: {MAE_SolverValues[highest]*salary_sd}")
    myLog.indent(1, f"Lowest Value of MSE: {MSE_SolverValues[highest]*salary_sd}")
    myLog.indent(1, f"Lowest Value of MSE: {np.sqrt(MSE_SolverValues[highest])*salary_sd}")
    myLog.indent(1, f"Best Tol: {TOL[highest//10]}")
    myLog.indent(1, f"Best Number of Iter: {range(1,1_000,100)[(highest%10)//10]}")


    TOL = [1e-4, 5e-4, 1e-5]

    MAE_SolverValues = []
    MSE_SolverValues = []
    for tol in TOL:
        if verbose: 
            myLog.indent(2, f"For tol: {tol}")    
        mae_values = []
        mse_values = []
        for iter in range(1,5):
            if verbose: 
                myLog.indent(2, f'Testing Iter of: {iter}')
            mae_values.append(mae(LassoRegModel(xTrain, xTest, yTrain, yTest, tol, iter, True).predict(xTest), yTest))
            mse_values.append(mse(LassoRegModel(xTrain, xTest, yTrain, yTest, tol, iter, True).predict(xTest), yTest))
        MAE_SolverValues += mae_values
        MSE_SolverValues += mse_values
    highest = np.argmin(np.array(MAE_SolverValues))
    myLog.indent(1, f"Lowest Value of MAE: {MAE_SolverValues[highest]*salary_sd}")
    myLog.indent(1, f"Lowest Value of MSE: {MSE_SolverValues[highest]*salary_sd}")
    myLog.indent(1, f"Lowest Value of MSE: {np.sqrt(MSE_SolverValues[highest])*salary_sd}")
    print(highest)
    myLog.indent(1, f"Best Tol: {TOL[highest//10]}")
    myLog.indent(1, f"Best Number of Iter: {range(1,1_000,100)[(highest%10)]}")