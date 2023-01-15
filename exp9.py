from myLogger import myLog
from imports import *
from models import *

def EXP9(X, Y, verbose): 

    if verbose:
        myLog.heading("EXP 9: NEURAL NETWORK")

    model = Sequential()        # a model consisting of successive layers
    # input layer
    model.add(Dense(32, 
                    kernel_initializer='normal', 
                    activation='relu'))

    # # hidden layer
    # model.add(Dense(16, 
    #                 kernel_initializer='normal', 
    #                 activation='relu'))
    # model.add(Dense(4, 
    #                 kernel_initializer='normal', 
    #                 activation='relu'))
    # output layer, with one neuron
    model.add(Dense(1, kernel_initializer='normal'))
    # compile the model
    model.compile(loss='mean_absolute_error', optimizer=keras.optimizers.Adagrad(learning_rate=10))
    
    
    xTrain, xTest, yTrain, yTest = tts(X, Y, test_size=0.1, shuffle=True)

    model.fit(xTrain, yTrain, batch_size=64, epochs=10, validation_split=0.1)

    predictions = model.predict(xTest)


    myLog.indent(1, f"MAE IS: {mae(predictions, yTest)}")