from imports import *
from models import *
from preprocessing import *
from myLogger import myLog
from exp3 import EXP3
from exp4a import EXP4A

if __name__ == '__main__':
    # READING THE DATA
    df = pd.read_csv("./data/archive/ds_salaries.csv")
    df = df[['work_year', 'experience_level', 'job_title', 'salary_in_usd', 'company_location', 'employee_residence', 'company_size']]
    df.columns = ['year', 'exp', 'title', 'raw_salary','company_loc', 'emp_loc', 'company_size']

    # LOGGING
    logger = myLog()

    NUMERICAL_FEATURES = ['year', 'raw_salary']
    CATEGORICAL_FEATURES = ['exp', 'title','company_loc', 'company_size']

    EXP2 = False
    EXP4B = False

    # PREPROCESSING CLASS
    df = Preprocessor(df, verbose=EXP2).preproc().reset_index(drop=True)

    data_mean, data_std = np.mean(df['raw_salary']), np.std(df['raw_salary'])
    
    # IDENTIFYING AND REMOVING OUTLIERS
    cut_off = data_std * 2
    lower, upper = data_mean - cut_off, data_mean + cut_off
    df = df.loc[df['raw_salary'] < upper]
    nonoutliers = [x for x in df['raw_salary'] if x < upper]
    outliers = [x for x in df['raw_salary'] if x > upper]
    Y = df[['raw_salary']]
    X = df.drop(['raw_salary'], axis=1)

    # MODEL SELECTION
    EXP3(X, Y, True)

    # POLY SELECTION
    EXP4A(X, Y, True)
    
    # LINEAR SELECTION
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