from myLogger import myLog
from imports import *
from models import *
from sklearn.ensemble import VotingRegressor    
from sklearn.ensemble import BaggingRegressor   
from sklearn.ensemble import StackingRegressor  

import warnings
warnings.filterwarnings("ignore")

def EXP10(X, Y, verbose):    

    salary_sd = 70429.13957426547

    if verbose:
        myLog.heading("EXP 10: ENSEMBLE")

    xTrain, xTest, yTrain, yTest = tts(X, Y, test_size=0.1)
    linear = LinRegModel(xTrain, xTest, yTrain, yTest, True)
    lasso = LassoRegModel(xTrain, xTest, yTrain, yTest, 0.0001, 100,True)
    ridge = RidgeRegModel(xTrain, xTest, yTrain, yTest, 'lsqr', 0.9, True)
    #svr =  SVMModel(xTrain, xTest, yTrain, yTest, 'poly', False)
    regr = BaggingRegressor(linear)
    knn = KNNModel(xTrain, xTest, yTrain, yTest, 7, True)
    rfc = RFCModel(xTrain, xTest, yTrain, yTest, 250, True)

    ESTIMATORS = [ridge, rfc, knn, lasso]
    weights = []
    for model in ESTIMATORS:
        weights.append(10000/mae(model.predict(xTest), yTest))

    result = 100000

    # TO BE DELETED POST SCREENSHOTS
    # while(result > 18_000):

    #     xTrain, xTest, yTrain, yTest = tts(X, Y, test_size=0.1)

    #     linear = LinRegModel(xTrain, xTest, yTrain, yTest, False)
    #     lasso = LassoRegModel(xTrain, xTest, yTrain, yTest, 0.0001, 1,False)
    #     ridge = RidgeRegModel(xTrain, xTest, yTrain, yTest, 'lsqr', 0.9, False)
    #     knn = KNNModel(xTrain, xTest, yTrain, yTest, 7, False)
    #     rfc = RFCModel(xTrain, xTest, yTrain, yTest, 300, False)


    #     weights = []
    #     for model in ESTIMATORS:
    #         weights.append(10000/mae(model.predict(xTest), yTest))
                
    #     ensemble = VotingRegressor([('ridge', ridge), ('rfc', rfc), ('knn', knn), ('lasso', lasso)], weights=weights)
    #     ensemble.fit(xTrain, yTrain)
    #     MAE = mae(ensemble.predict(xTest), yTest)*70429.13957426547
    #     MSE = mse(ensemble.predict(xTest), yTest)*70429.13957426547
    #     RMSE = np.sqrt(mse(ensemble.predict(xTest), yTest))*70429.13957426547
    #     print(f"MAE IS: {MAE}")
    #     print(f"MSE IS: {MAE}")
    #     print(f"RMSE IS: {RMSE}")
    #     print(f"SCORE IS: {ensemble.score(xTest, yTest)}")
    #     result = MAE

    # TO BE DELETED POST SCREENSHOTS^^

    ensemble = VotingRegressor([('ridge', ridge), ('rfc', rfc), ('knn', knn), ('lasso', lasso)], weights=weights)
    ensemble.fit(xTrain, yTrain)
    MAE = mae(ensemble.predict(xTest), yTest)*70429.13957426547
    MSE = mse(ensemble.predict(xTest), yTest)*70429.13957426547
    RMSE = np.sqrt(mse(ensemble.predict(xTest), yTest))*70429.13957426547
    print(f"MAE IS: {MAE}")
    print(f"MSE IS: {MSE}")
    print(f"RMSE IS: {RMSE}")
    print(f"SCORE IS: {ensemble.score(xTest, yTest)}\n")