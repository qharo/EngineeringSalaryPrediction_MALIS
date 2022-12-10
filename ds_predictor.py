import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor as rfr


df = pd.read_csv("./data/archive/ds_salaries.csv")
# df = df[['work_year', 'experience_level', 'job_title', 'salary_in_usd', 'employee_residence', 'company_location', 'company_size']]
# df.columns = ['raw_year', 'exp', 'title', 'raw_salary', 'emp_loc', 'company_loc', 'company_size']
df = df[['work_year', 'experience_level', 'job_title', 'salary_in_usd', 'company_location', 'employee_residence', 'company_size']]
df.columns = ['year', 'exp', 'title', 'raw_salary','company_loc', 'emp_loc', 'company_size']



def onehenc(df, colName):
    oneh = pd.get_dummies(df[colName], prefix=colName)
    print(" Performing One-Hot Encoding for: "+colName)
    print(f" Shape before OHE: {df.shape}")
    df = pd.concat([df, oneh], axis=1)
    df = df.drop([colName], axis=1)
    print(f" Shape after OHE: {df.shape} \n")
    return df

def zscale(df, col, col2):
    zscaler = StandardScaler()

    zscaled = pd.DataFrame(zscaler.fit_transform(df[col]))


    zscaled.columns = col2

    df = pd.concat([df, zscaled], axis=1)

    df = df.drop(col, axis=1)

    print(zscaler.mean_, np.sqrt(zscaler.var_))

    return df, zscaler.mean_, np.sqrt(zscaler.var_)

def preproc(df):
    print("\n ######### PREPROCESSING ######### ")

    print(df.columns)
    df = df.reset_index(drop=True)

    print(" ------- CATEGORICAL DATA ------- ")
    for col in ['title', 'company_loc', 'exp', 'company_size', 'emp_loc']:
        df = onehenc(df, col)


    # df = df.reset_index(drop=True)
    print(" ------- NUMERICAL DATA ------- ")
    df, mean, std = zscale(df, ['raw_salary'], ['salary'])
    #for col,col2 in zip(['raw_year', 'raw_salaries'], ['year', 'salaries']):
    #    df = zscale(df, col, col2)

    return df, mean, std
 
def LinRegModel(xTrain, xTest, yTrain, yTest):
    model = LinearRegression()
    model.fit(xTrain, yTrain)
    yhat = model.predict(xTest)
    #print(yhat)
    print(f" MSE is: {mse(yhat, yTest)}")

def LogRegModel(xTrain, xTest, yTrain, yTest):
    model = LogisticRegression()
    model.fit(xTrain, yTrain)
    yhat = model.predict(xTest)
    #print(yhat)
    print(f" MSE is: {mse(yhat, yTest)}")

def SVMModel(xTrain, xTest, yTrain, yTest):
    model = SVR(kernel='poly', degree=10)
    model.fit(xTrain, yTrain)
    yhat = model.predict(xTest)
    #print(yhat)
    print(f" MSE is: {mse(yhat, yTest)}")
    print(f" MAE is {mae(yhat, yTest)}")
    print(f" Score is: {model.score(xTest, yTest)}")
    return model, mse(yhat, yTest, squared=True)

def RFCModel(xTrain, xTest, yTrain, yTest):
    model = rfr()
    model.fit(xTrain, yTrain)
    yhat = model.predict(xTest)
    #print(yhat)
    print(f" MSE is: {mse(yhat, yTest, squared=False)}")
    print(f" MAE is {mae(yhat, yTest)}")
    print(f" Score is: {model.score(xTest, yTest)}")
    return model, mae(yhat, yTest)

if __name__ == '__main__':
    print(df.columns)


    print(f"CORRELATION: {df['raw_salary'].corr(df['year'])}")
    #print(f"CORRELATION: {df['raw_salary'].corr(df['title'])}")
    #print(f"CORRELATION: {df['raw_salary'].corr(df['year'])}")
    #print(f"CORRELATION: {df['raw_salary'].corr(df['year'])}")

    # YEAR
    # print(20*"-" + " Analyzing Year".center(20) + 20*"-")
    # print(df['raw_year'].value_counts())
    # print(np.mean(df[df['raw_year'] == 2020]['raw_salary']))
    # print(np.mean(df[df['raw_year'] == 2021]['raw_salary']))
    # print(np.mean(df[df['raw_year'] == 2022]['raw_salary']))
    df['year'] = df['year'] - 2021

    # EXPERIENCE
    # print(20*"-" + " Analyzing Year".center(20) + 20*"-")
    # print(np.mean(df[df['exp'] == 'SE']['raw_salary']))
    # print(np.mean(df[df['exp'] == 'MI']['raw_salary']))
    # print(np.mean(df[df['exp'] == 'EN']['raw_salary']))
    # print(np.mean(df[df['exp'] == 'EX']['raw_salary']))
    # print(df['exp'].value_counts())

    # TITLE
    # print(df['title'].value_counts())
    #df = df[df['title'].map(df['title'].value_counts()) > 20]
    # print(df.shape)
    # for title in df['title'].unique():
    #     print(np.mean(df[df['title'] == title]['raw_salary']))

    # COMPANY LOCATION
    df = df[df['company_loc'].map(df['company_loc'].value_counts()) > 20]
    # print(df['company_location'].value_counts())
    # print(df.groupby('company_location').mean())
    # for title in df['company_location'].unique():
    #     print(f"{title}: {np.mean(df[df['company_location'] == title]['salary_in_usd'])}")

    # COMPANY SIZE
    #df = df[df['company_location'].map(df['company_location'].value_counts()) > 20]
    # print(df['company_size'].value_counts())
    # print(df.groupby('company_size').mean())
    #for title in df['company_location'].unique():
    #    print(f"{title}: {np.mean(df[df['company_location'] == title]['salary_in_usd'])}")

    # print(np.mean(df[df['title'] == 'Data Scientist']['raw_salary']))
    # print(np.mean(df[df['title'] == 'Data Engineer']['raw_salary']))
    # print(np.mean(df[df['title'] == 'Data Analyst']['raw_salary']))
    # print(np.mean(df[df['title'] == 'Machine Learning Engineer']['raw_salary']))
    # print(np.mean(df[df.query("title not in ['Machine Learning Engineer', 'Data Analyst', 'Data Engineer', 'Data Scientist']")]['raw_salary']))


    # df = df[df['title'].map(df['title'].value_counts()) > 2]
    #df['raw_salary'] = np.abs(df['raw_salary']*100)
    print(df['raw_salary'])
    
    df, mean, std = preproc(df)

    # # SPLIT DATA
    Y = df[['salary']]
    X = df.drop(['salary'], axis=1)
    #print(df['company_loc'].value_counts())
    xTrain, xTest, yTrain, yTest = tts(X, Y, test_size=0.2, stratify=df['year'])




   # print(xTrain.shape, xTest.shape)

    # print(xTrain.isna().sum())
    # for col in ['title', 'company_loc', 'emp_loc', 'exp', 'company_size']:
    #     xTrain = onehenc(xTrain, col)
    #     xTest = onehenc(xTest, col)

    # xTrain = xTrain.reset_index(drop=True)
    # xTrain = zscale(xTrain, ['raw_year'], ['year'])
    # xTest = xTest.reset_index(drop=True)
    # xTest = zscale(xTest, ['raw_year'], ['year'])

    # yTrain = yTrain.reset_index(drop=True)
    # yTrain = zscale(yTrain, ['raw_salary'], ['salary'])
    # yTest = yTest.reset_index(drop=True)
    # #print(yTest.reset_index(drop=True))
    # yTest = zscale(yTest, ['raw_salary'], ['salary'])



    # xTrain = preproc(X, ['title', 'company_loc', 'emp_loc', 'exp', 'company_size'], ['raw_year'], ['year'])
    # xTest = preproc(X, ['title', 'company_loc', 'emp_loc', 'exp', 'company_size'], ['raw_year'], ['year'])
    # yTrain = preproc(Y, [], ['raw_salary'], ['salary'])
    # yTest = preproc(Y, [], ['raw_salary'], ['salary'])

    LinRegModel(xTrain, xTest, yTrain, yTest)
    #LogRegModel(xTrain, xTest, yTrain, yTest)
    svmModel, msesvm = SVMModel(xTrain, xTest, yTrain, yTest)
    rfrModel, mserfc = RFCModel(xTrain, xTest, yTrain, yTest)
    print(f"Off by {msesvm*std+mean}")





# BASE MODEL IS ONE HOT ENCODING, K FOLDS VALIDATION AND SPLIT TRAIN TEST