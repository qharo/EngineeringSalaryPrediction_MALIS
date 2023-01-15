from myLogger import myLog
from imports import *
from models import *
from sklearn.ensemble import VotingRegressor    
from sklearn.ensemble import BaggingRegressor   
from sklearn.ensemble import StackingRegressor  

import warnings
warnings.filterwarnings("ignore")

def EXP10(X, Y, verbose):    
    if verbose:
        myLog.heading("EXP 10: ENSEMBLE")
        #myLog.indent(1, "4 models to consider: Linear Regression, SVM (Poly), SVM (RBF), SVM (Linear)")
    
    verbose2 = False
    verbose3 = False
    # kfolds = kf(n_splits=4, shuffle=True)
    test_value = 30_000
    nrounds = 0
    results = []
    # xTrain, xTest, yTrain, yTest = tts(X, Y, test_size=0.2, shuffle=True)
    while ((test_value > 20_000) and (nrounds < 51)):
        xTrain, xTest, yTrain, yTest = tts(X, Y, test_size=0.2, shuffle=True)
        svr, svr_preds =  SVMModel(xTrain, xTest, yTrain, yTest, 'poly', verbose2)
        # regr = BaggingRegressor(linear)
        knn, knn_preds = KNNModel(xTrain, xTest, yTrain, yTest, verbose2)
        rfc, rfc_preds = RFCModel(xTrain, xTest, yTrain, yTest, 500, verbose2)
        linear, lr_preds = LinRegModel(xTrain, xTest, yTrain, yTest, verbose3)
        lasso, lasso_preds = LassoRegModel(xTrain, xTest, yTrain, yTest, 10, verbose3)
        ridge, ridge_preds = RidgeRegModel(xTrain, xTest, yTrain, yTest, 'lsqr', 0, verbose3)
        ensemble = VotingRegressor([('lasso', lasso), ('linear', linear), ('ridge', ridge)])
        ensemble.fit(xTrain, yTrain)
        yHat = ensemble.predict(xTest)
        val_mae = mae(yHat, yTest)
        results.append(val_mae)
        nrounds += 1
        test_value = np.average(results)
        print(f"{svr.score(xTest, yTest)}\t{knn.score(xTest, yTest)}\t{rfc.score(xTest, yTest)}\t{linear.score(xTest, yTest)}\t{lasso.score(xTest, yTest)}\t{ridge.score(xTest, yTest)}\t{ensemble.score(xTest, yTest)}")
        # print(test_value)
        # print(f"MAE IS: {test_value}")
        # print(f"SCORE IS: {ensemble.score(xTest, yTest)}")




    # train_indices = []
    # test_indices = []

    # for train_index, test_index in kfolds.split(X):
    #     train_indices.append(train_index)
    #     test_indices.append(test_index)

    
    # xTrain, xTest, yTrain, yTest = indexToSplit(X, Y, train_indices[0], test_indices[0])

    # values = []
    # for iter in range(1,1_000, 100):
    #     if verbose: 
    #         myLog.indent(2, f'Iterations: {iter}')
    #     values.append(LassoRegModel(xTrain, xTest, yTrain, yTest, 100, True))
    # print(np.argmax(values))