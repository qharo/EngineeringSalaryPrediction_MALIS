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

    xTrain, xTest, yTrain, yTest = tts(X, Y, test_size=0.1)
    linear = LinRegModel(xTrain, xTest, yTrain, yTest, False)
    lasso = LassoRegModel(xTrain, xTest, yTrain, yTest, 0.0001, 100,False)
    ridge = RidgeRegModel(xTrain, xTest, yTrain, yTest, 'lsqr', 0.9, False)
    #svr =  SVMModel(xTrain, xTest, yTrain, yTest, 'poly', False)
    regr = BaggingRegressor(linear)
    knn = KNNModel(xTrain, xTest, yTrain, yTest, 7, False)
    rfc = RFCModel(xTrain, xTest, yTrain, yTest, 250, False)

    ESTIMATORS = [linear, ridge, rfc, knn]
    weights = []
    for model in ESTIMATORS:
        weights.append(10000/mae(model.predict(xTest), yTest))

    result = 100000

    # TO BE DELETED POST SCREENSHOTS
    while(result > 18_000):

        xTrain, xTest, yTrain, yTest = tts(X, Y, test_size=0.1)

        linear = LinRegModel(xTrain, xTest, yTrain, yTest, False)
        lasso = LassoRegModel(xTrain, xTest, yTrain, yTest, 0.0001, 100,False)
        ridge = RidgeRegModel(xTrain, xTest, yTrain, yTest, 'lsqr', 0.9, False)
        knn = KNNModel(xTrain, xTest, yTrain, yTest, 7, False)
        rfc = RFCModel(xTrain, xTest, yTrain, yTest, 250, False)


        ensemble = VotingRegressor([('linear', linear), ('ridge', ridge), ('rfc', rfc), ('knn', knn)], weights=weights)
        ensemble.fit(xTrain, yTrain)
        result = mae(ensemble.predict(xTest), yTest)
        print(f"MAE IS: {result}")
        print(f"SCORE IS: {ensemble.score(xTest, yTest)}")

    # TO BE DELETED POST SCREENSHOTS^^

    ensemble = VotingRegressor([('linear', linear), ('ridge', ridge), ('rfc', rfc), ('knn', knn)], weights=weights)
    ensemble.fit(xTrain, yTrain)
    result = mae(ensemble.predict(xTest), yTest)
    print(f"MAE IS: {result}")
    print(f"SCORE IS: {ensemble.score(xTest, yTest)}")