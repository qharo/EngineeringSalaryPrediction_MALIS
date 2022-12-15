from imports import *
from models import *
from preprocessing import *

def indexToSplit(X, Y, train_index, test_index):
    xTrain = X.iloc[train_index,:]
    yTrain = Y.iloc[train_index,:]
    xTest = X.iloc[test_index,:]
    yTest = Y.iloc[test_index,:]
    return xTrain, xTest, yTrain, yTest

if __name__ == '__main__':
    df = pd.read_csv("./data/archive/ds_salaries.csv")
    df = df[['work_year', 'experience_level', 'job_title', 'salary_in_usd', 'company_location', 'employee_residence', 'company_size']]
    df.columns = ['year', 'exp', 'title', 'raw_salary','company_loc', 'emp_loc', 'company_size']

    NUMERICAL_FEATURES = ['year', 'raw_salary']
    CATEGORICAL_FEATURES = ['exp', 'title','company_loc', 'company_size']

    EXP2 = False
    EXP3 = False
    EXP4A = False
    EXP4B = True

    # PASS TRUE FOR EXPERIMENT 2
    df = Preprocessor(df, verbose=EXP2).preproc().reset_index(drop=True)

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
