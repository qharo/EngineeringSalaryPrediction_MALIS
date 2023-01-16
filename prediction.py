from imports import *
from models import *
from preprocessing import *
from myLogger import myLog
from exp3 import EXP3
from exp4 import EXP4
from exp5 import EXP5
from exp6 import EXP6
from exp7 import EXP7
from exp8 import EXP8
from exp9 import EXP9
from exp10 import EXP10
from visualizer import visualize

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
    Prep = Preprocessor(df, verbose=EXP2)
    df = Prep.preproc().reset_index(drop=True)

    salary_avg = Prep.salary_avg
    salary_sd = Prep.salary_sd

    #print(salary_sd, salary_avg)

    data_mean, data_std = np.mean(df['raw_salary']), np.std(df['raw_salary'])
    

    # IDENTIFYING AND REMOVING OUTLIERS
    cut_off = data_std * 2
    lower, upper = data_mean - cut_off, data_mean + cut_off
    df = df.loc[df['raw_salary'] < upper]
    nonoutliers = [x for x in df['raw_salary'] if x < upper]
    outliers = [x for x in df['raw_salary'] if x > upper]
    Y = df[['raw_salary']]
    X = df.drop(['raw_salary'], axis=1)
    
    EXP3(X, Y, True)
    EXP4(X, Y, True) # LINEAR AND POLY SELECTION
    EXP5(X, Y, True) # RFC TUNING
    EXP6(X, Y, True) # KNN TUNING
    EXP7(X, Y, True) # RIDGE REGRESSION TUNING
    EXP8(X, Y, True) # LASSO REGRESSION TUNING
    EXP9(X, Y, True) # NEURAL NETWORK
    EXP10(X, Y, True) # ENSEMBLE

