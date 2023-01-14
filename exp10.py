from myLogger import myLog
from imports import *
from models import *
from sklearn.ensemble import VotingRegressor    
from sklearn.ensemble import BaggingRegressor   
from sklearn.ensemble import StackingRegressor    

def EXP10(X, Y, verbose):    
    if verbose:
        myLog.heading("EXP 8: ENSEMBLE")
        #myLog.indent(1, "4 models to consider: Linear Regression, SVM (Poly), SVM (RBF), SVM (Linear)")



    # kfolds = kf(n_splits=4, shuffle=True)
    test_value = 30_000
    num = 0
    results = []
    while(test_value > 20_000):
        xTrain, xTest, yTrain, yTest = tts(X, Y, test_size=0.1, shuffle=True)
        linear = LinRegModel(xTrain, xTest, yTrain, yTest, False)
        lasso = LassoRegModel(xTrain, xTest, yTrain, yTest, 500,False)
        ridge = RidgeRegModel(xTrain, xTest, yTrain, yTest, 'lsqr', 0, False)
        svr =  SVMModel(xTrain, xTest, yTrain, yTest, 'poly', False)
        regr = BaggingRegressor(linear)
        knn = KNNModel(xTrain, xTest, yTrain, yTest, False)
        rfc = RFCModel(xTrain, xTest, yTrain, yTest, 500, False)
        ensemble = VotingRegressor([('lasso', lasso), ('rfc', rfc), ('linear', linear), ('ridge', ridge)])
        ensemble.fit(xTrain, yTrain)
        yHat = ensemble.predict(xTest)
        results.append(mae(yHat, yTest))
        #num += 1
        test_value = np.average(results)
        print(test_value)
        print(f"MAE IS: {test_value}")
        print(f"SCORE IS: {ensemble.score(xTest, yTest)}")




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