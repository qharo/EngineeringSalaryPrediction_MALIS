import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as tts, StratifiedKFold as skf, KFold as kf
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelBinarizer, KBinsDiscretizer, OrdinalEncoder
from sklearn_pandas import DataFrameMapper
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline as make
from sklearn.impute import SimpleImputer
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from keras.models import Sequential
from keras.layers import Dense 
import keras
import pickle

def indexToSplit(X, Y, train_index, test_index):
    xTrain = X.iloc[train_index,:]
    yTrain = Y.iloc[train_index,:]
    xTest = X.iloc[test_index,:]
    yTest = Y.iloc[test_index,:]
    return xTrain, xTest, yTrain, yTest