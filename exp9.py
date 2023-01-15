from myLogger import myLog
from imports import *
from models import *

def EXP9(X, Y, verbose): 
    model = Sequential()        # a model consisting of successive layers
    # input layer
    model.add(Dense(10, 
                    kernel_initializer='normal', 
                    activation='relu'))
    model.add(Dense(4, 
                    kernel_initializer='normal', 
                    activation='relu'))
    # output layer, with one neuron
    model.add(Dense(1, kernel_initializer='normal'))
    # compile the model
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    
    xTrain, xTest, yTrain, yTest = tts(X, Y, test_size=0.1, shuffle=True)

    model.fit(xTrain, yTrain)

    predictions = model.predict(xTest)


    print(f"MAE IS: {mae(predictions, yTest)}")
    # print(f"SCORE IS: {ensemble.score(xTest, yTest)}")
    print("test")

    print(model)