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