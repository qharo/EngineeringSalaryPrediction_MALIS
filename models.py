from imports import *
from myLogger import myLog

def LinRegModel(xTrain, xTest, yTrain, yTest, verbose):
    model = LinearRegression()
    model.fit(xTrain, yTrain)
    yhat = model.predict(xTest)    
    if verbose:
        myLog.indent(2, 'LINEAR REGRESSION')
        #print("\n" + 20*"-" + " Linear Reg ".center(20) + 20*"-")
        myLog.indent(3, f"MSE is: {mse(yhat, yTest)}")
        myLog.indent(3, f"MAE is {mae(yhat, yTest)}")
        myLog.indent(3, f"Score is: {model.score(xTest, yTest)}")
    return model


def LassoRegModel(xTrain, xTest, yTrain, yTest, verbose):
    model = Lasso()
    model.fit(xTrain, yTrain)
    yhat = model.predict(xTest)    
    if verbose:
        print("\n" + 20*"-" + " Lasso Reg ".center(20) + 20*"-")
        print(f" MSE is: {mse(yhat, yTest)}")
        print(f" MAE is {mae(yhat, yTest)}")
        print(f" Score is: {model.score(xTest, yTest)}\n")
    return model


def RidgeRegModel(xTrain, xTest, yTrain, yTest, verbose):
    model = Ridge()
    model.fit(xTrain, yTrain)
    yhat = model.predict(xTest)    
    if verbose:
        print("\n" + 20*"-" + " Ridge Reg ".center(20) + 20*"-")
        print(f" MSE is: {mse(yhat, yTest)}")
        print(f" MAE is {mae(yhat, yTest)}")
        print(f" Score is: {model.score(xTest, yTest)}\n")
    return model

def SVMModel(xTrain, xTest, yTrain, yTest, kernel, verbose, deg=5):
    model = SVR(kernel=kernel, degree=deg)

    # print(yTrain)

    model.fit(xTrain, yTrain)
    yhat = model.predict(xTest)
    if verbose:
        myLog.indent(2, 'SVM: LINEAR')
        myLog.indent(3, f"MSE is: {mse(yhat, yTest)}")
        myLog.indent(3, f"MAE is {mae(yhat, yTest)}")
        myLog.indent(3, f"Score is: {model.score(xTest, yTest)}\n")
    return model, mse(yhat, yTest, squared=True)