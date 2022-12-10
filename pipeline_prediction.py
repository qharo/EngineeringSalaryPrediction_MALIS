import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline, make_pipeline as make
from sklearn_pandas import DataFrameMapper
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelBinarizer, KBinsDiscretizer, OrdinalEncoder
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split as tts, StratifiedKFold as skf, KFold as kf
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae
from sklearn.linear_model import LinearRegression, Lasso, Ridge

def binning(df, n):
    mini = min(df)-1
    maxi = max(df)+1
    bins = np.arange(mini, maxi, (maxi-mini)/n)
    print(f" SIZE OF EACH BIN: {(maxi-mini)/n}")
    df = pd.cut(df, bins=bins, labels=range(len(bins)-1))
    #df = pd.qcut(df, q=n, labels=range(n))
    return df

def preproc(df, verbose, CATEGORICAL_FEATURES, NUMERICAL_FEATURES):

    if verbose:
        # ANALYZING YEAR
        print("\n"+40*"#" + " EXP 2: FEATURE SELECTION ".center(20) + 40*"#")
        print(20*"-" + " Analyzing Year ".center(20) + 20*"-")
        for title in df['year'].unique():
            print(f" Mean of the year {title}: {np.mean(df[df['year'] == title]['raw_salary'])}")
        print("\n")

        # ANALYZING YEAR
        print(20*"-" + " Analyzing Exp ".center(20) + 20*"-")
        for title in df['exp'].unique():
            print(f" Mean of the Experience Level {title}: {np.mean(df[df['exp'] == title]['raw_salary'])}")
        print("\n")


        # ANALYZING TITLE
        print(20*"-" + " Analyzing Title ".center(20) + 20*"-")
        for title in df['title'].unique():
            print(f" Mean of the Experience Level {title}: {np.mean(df[df['title'] == title]['raw_salary'])} Count: {df[df['title'] == title].shape[0]}")
        print(" Working with over 50 unique titles will not yield good results, especially when they're unequally distributed. Thus, we selected those that occur more than 20 times.")
        df = df[df['title'].map(df['title'].value_counts()) > 20]
        print("\n")
        for title in df['title'].unique():
            print(f" Mean of the Experience Level {title}: {np.mean(df[df['title'] == title]['raw_salary'])} Count: {df[df['title'] == title].shape[0]}")    
        print("\n")

        # ANALYZING COMPANY LOCATION
        print(20*"-" + " Analyzing Company Loc ".center(20) + 20*"-")
        for title in df['company_loc'].unique():
            print(f" Mean of the Company Loc {title}: {np.mean(df[df['company_loc'] == title]['raw_salary'])} Count: {df[df['company_loc'] == title].shape[0]}")
        print(" Working with over 50 unique company locations will not yield good results, especially when they're unequally distributed. Thus, we selected those that occur more than 15 times.")
        df = df[df['company_loc'].map(df['company_loc'].value_counts()) > 15]
        print("\n")
        for title in df['company_loc'].unique():
            print(f" Mean of the Company Loc {title}: {np.mean(df[df['company_loc'] == title]['raw_salary'])} Count: {df[df['company_loc'] == title].shape[0]}")    
        print("\n")


        # ANALYZING EMP LOCATION
        print(20*"-" + " Analyzing Employee Loc ".center(20) + 20*"-")
        for title in df['emp_loc'].unique():
            print(f" Mean of the Emp Loc {title}: {np.mean(df[df['emp_loc'] == title]['raw_salary'])} Count: {df[df['emp_loc'] == title].shape[0]}")
        print(" Working with over 20 unique emp locations will not yield good results, especially when they're unequally distributed. Thus, we selected those that occur more than 10 times.")
        df = df[df['emp_loc'].map(df['emp_loc'].value_counts()) > 10]
        print("\n")
        for title in df['emp_loc'].unique():
            print(f" Mean of the Emp Loc {title}: {np.mean(df[df['emp_loc'] == title]['raw_salary'])} Count: {df[df['emp_loc'] == title].shape[0]}")    
        print("\n")


        # ANALYZING COMPANY SIZE
        print(20*"-" + " Analyzing Company Size ".center(20) + 20*"-")
        for title in df['company_size'].unique():
            print(f" Mean of the Company Size {title}: {np.mean(df[df['company_size'] == title]['raw_salary'])}")
        print("\n")


        print(100*"-" + "\n")
        print(" We found that for Company Size and Experiene, Ordinal Encoding makes sense as there is a progression, whilst Title and Company Location is OneHotEncoded.")

        print("\n" + 100*"-")

    else:
        df = df[df['title'].map(df['title'].value_counts()) > 20]        
        df = df[df['company_loc'].map(df['company_loc'].value_counts()) > 15]    
        df = df[df['emp_loc'].map(df['emp_loc'].value_counts()) > 10]


    #df['raw_salary'] = binning(df['raw_salary'], 10)

    TRANSFORMS = [(value, [LabelBinarizer()]) for value in ['title', 'company_loc', 'emp_loc']]
    TRANSFORMS += [(['exp'], [OrdinalEncoder()])]
    #TRANSFORMS += [(['raw_salary'], [StandardScaler()])]
    TRANSFORMS += [(['raw_salary'], [SimpleImputer()])]
    TRANSFORMS += [(['year'], OrdinalEncoder())]

    mapper = DataFrameMapper(TRANSFORMS, df_out=True)
    return mapper.fit_transform(df)

def indexToSplit(X, Y, train_index, test_index):
    xTrain = X.iloc[train_index,:]
    yTrain = Y.iloc[train_index,:]
    xTest = X.iloc[test_index,:]
    yTest = Y.iloc[test_index,:]
    return xTrain, xTest, yTrain, yTest

def LinRegModel(xTrain, xTest, yTrain, yTest, verbose):
    model = LinearRegression()
    model.fit(xTrain, yTrain)
    yhat = model.predict(xTest)    
    if verbose:
        print("\n" + 20*"-" + " Linear Reg ".center(20) + 20*"-")
        print(f" MSE is: {mse(yhat, yTest)}")
        print(f" MAE is {mae(yhat, yTest)}")
        print(f" Score is: {model.score(xTest, yTest)}\n")
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

def SVMModel(xTrain, xTest, yTrain, yTest, verbose, kernel, deg=None):
    model = SVR(kernel=kernel, degree=deg)
    model.fit(xTrain, yTrain)
    yhat = model.predict(xTest)
    if verbose:
        print("\n" + 20*"-" + " SVR ".center(20) + 20*"-")
        print(f" MSE is: {mse(yhat, yTest)}")
        print(f" MAE is {mae(yhat, yTest)}")
        print(f" Score is: {model.score(xTest, yTest)}\n")
    return model, mse(yhat, yTest, squared=True)

if __name__ == '__main__':
    df = pd.read_csv("./data/archive/ds_salaries.csv")
    df = df[['work_year', 'experience_level', 'job_title', 'salary_in_usd', 'company_location', 'employee_residence', 'company_size']]
    df.columns = ['year', 'exp', 'title', 'raw_salary','company_loc', 'emp_loc', 'company_size']

    NUMERICAL_FEATURES = ['year', 'raw_salary']
    CATEGORICAL_FEATURES = ['exp', 'title','company_loc', 'company_size']

    EXP2 = False
    EXP3 = False
    EXP4A = True
    EXP4B = False

    # PASS TRUE FOR EXPERIMENT 2
    df = preproc(df, EXP2, CATEGORICAL_FEATURES, NUMERICAL_FEATURES).reset_index(drop=True)

    #print

    data_mean, data_std = np.mean(df['raw_salary']), np.std(df['raw_salary'])
    # identify outliers
    cut_off = data_std * 2
    lower, upper = data_mean - cut_off, data_mean + cut_off
    # print(lower, upper)
    # print(df.shape)
    df = df.loc[df['raw_salary'] < upper]
    # print(df.shape)

    nonoutliers = [x for x in df['raw_salary'] if x < upper]
    outliers = [x for x in df['raw_salary'] if x > upper]
    # print(len(outliers))
    # print(len(nonoutliers)) 

   # print(df['raw_salary'].head())

    Y = df[['raw_salary']]
    X = df.drop(['raw_salary'], axis=1)

    if EXP3:
        print("\n"+40*"#" + " EXP 3: MODEL SELECTION ".center(20) + 40*"#")
        print(" We shall consider 4 models: Linear Regression and SVM with 3 kernels (linear, rbf, poly).")
        kfolds = kf(n_splits=4, shuffle=True)

        train_indices = []
        test_indices = []

        for train_index, test_index in kfolds.split(X):
            train_indices.append(train_index)
            test_indices.append(test_index)
        
        xTrain, xTest, yTrain, yTest = indexToSplit(X, Y, train_indices[0], test_indices[0])
        LinRegModel(xTrain, xTest, yTrain, yTest, EXP3)
        xTrain, xTest, yTrain, yTest = indexToSplit(X, Y, train_indices[1], test_indices[1])
        print(" Kernel: Linear")
        SVMModel(xTrain, xTest, yTrain, yTest, EXP3, 'linear')
        xTrain, xTest, yTrain, yTest = indexToSplit(X, Y, train_indices[2], test_indices[2])
        print(" Kernel: RBF")
        SVMModel(xTrain, xTest, yTrain, yTest, EXP3, 'rbf')
        xTrain, xTest, yTrain, yTest = indexToSplit(X, Y, train_indices[3], test_indices[3])
        print(" Kernel: Poly")
        SVMModel(xTrain, xTest, yTrain, yTest, EXP3, 'poly')

        print(" We find that the Linear Regression Model performs well. The SVM also performs well, especially the linear and poly kernel.")



    if EXP4A:
        print("\n"+40*"#" + " EXP 3: POLY SELECTION ".center(20) + 40*"#")
        print(" We shall consider 3 degrees: 2, 7, 11")
        kfolds = kf(n_splits=3, shuffle=True)

        train_indices = []
        test_indices = []

        for train_index, test_index in kfolds.split(X):
            train_indices.append(train_index)
            test_indices.append(test_index)
        
        xTrain, xTest, yTrain, yTest = indexToSplit(X, Y, train_indices[0], test_indices[0])
        SVMModel(xTrain, xTest, yTrain, yTest, EXP4A, 'poly', 2)
        xTrain, xTest, yTrain, yTest = indexToSplit(X, Y, train_indices[1], test_indices[1])
        SVMModel(xTrain, xTest, yTrain, yTest, EXP4A, 'poly', 10)
        xTrain, xTest, yTrain, yTest = indexToSplit(X, Y, train_indices[2], test_indices[2])
        SVMModel(xTrain, xTest, yTrain, yTest, EXP4A, 'poly', 20)
        # xTrain, xTest, yTrain, yTest = indexToSplit(X, Y, train_indices[3], test_indices[3])
        # print(" Kernel: Poly")
        # SVMModel(xTrain, xTest, yTrain, yTest, EXP3, 'poly')

        #print(" We find that the Linear Regression Model performs well. The SVM also performs well, especially the linear and poly kernel.")
        #print(" Based on this, we can conclude that SVM (linear) is the superior choice for this task.\n") 
        #    
    if EXP4B:
        print("\n"+40*"#" + " EXP 3: LINEAR SELECTION ".center(20) + 40*"#")
        print(" We shall consider 3 types of Linear Regressors: Linear, Lasso and Ridge")
        kfolds = kf(n_splits=3, shuffle=True)

        train_indices = []
        test_indices = []

        for train_index, test_index in kfolds.split(X):
            train_indices.append(train_index)
            test_indices.append(test_index)
        
        xTrain, xTest, yTrain, yTest = indexToSplit(X, Y, train_indices[0], test_indices[0])
        LinRegModel(xTrain, xTest, yTrain, yTest, EXP4B)
        xTrain, xTest, yTrain, yTest = indexToSplit(X, Y, train_indices[1], test_indices[1])
        LassoRegModel(xTrain, xTest, yTrain, yTest, EXP4B)
        xTrain, xTest, yTrain, yTest = indexToSplit(X, Y, train_indices[2], test_indices[2])
        RidgeRegModel(xTrain, xTest, yTrain, yTest, EXP4B)
    
    #model = SVMModel()
            # xTrain = X.iloc[train_indices,:]
            # yTrain = Y.iloc[train_indices,:]
            # xTest = X.iloc[test_indices,:]
            # yTest = Y.iloc[test_indices,:]

            # print('Fold',str(fold_no),'Class Ratio:',sum(test['Returned_Units'])/len(test['Returned_Units']))
            # fold_no += 1

        #sfk = Stratified
    


    #xTrain, xTest, yTrain, yTest = tts(X, Y, test_size=0.15, shuffle=True, stratify=df['year'])

    # linRegModel = LinRegModel(xTrain, xTest, yTrain, yTest, True)
    # svmModel, msesvm = SVMModel(xTrain, xTest, yTrain, yTest, True)   
