from myLogger import myLog
from imports import *
from models import *

def EXP9(X, Y, verbose): 

    salary_sd = 70429.13957426547   

    if verbose:
        myLog.heading("EXP 9: NEURAL NETWORK")

    model = Sequential()        # a model consisting of successive layers
    # input layer
    model.add(Dense(32, 
                    kernel_initializer='normal', 
                    activation='relu'))
    model.add(Dense(4, 
                    kernel_initializer='normal', 
                    activation='relu'))
    # output layer, with one neuron
    model.add(Dense(1, kernel_initializer='normal'))
    # compile the model
    model.compile(loss='mean_absolute_error', optimizer='sgd')
    
    
    xTrain, xTest, yTrain, yTest = tts(X, Y, test_size=0.1, shuffle=True)

    model.fit(xTrain, yTrain, epochs=100, validation_split=0.1)

    predictions = model.predict(xTest)


    myLog.indent(1, f"MAE IS: {mae(predictions, yTest)*salary_sd}")
    myLog.indent(1, f"MSE IS: {mse(predictions, yTest)*salary_sd}")
    myLog.indent(1, f"RMSE IS: {np.sqrt(mse(predictions, yTest))*salary_sd}")