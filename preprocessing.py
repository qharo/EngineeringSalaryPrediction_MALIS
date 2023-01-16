from imports import *

class Preprocessor:
    def __init__(self, df, verbose):
        self.df = df
        self.verbose = verbose
        self.salary_avg = 0
        self.salary_sd = 0

    def binning(df, n):
        mini = min(df)-1
        maxi = max(df)+1
        bins = np.arange(mini, maxi, (maxi-mini)/n)
        print(f" SIZE OF EACH BIN: {(maxi-mini)/n}")
        df = pd.cut(df, bins=bins, labels=range(len(bins)-1))
        #df = pd.qcut(df, q=n, labels=range(n))
        return df

    def preproc(self):
        df = self.df

        if self.verbose:
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
            # df = df[df['title'].map(df['title'].value_counts()) > 20]        
            df = df[df['company_loc'].map(df['company_loc'].value_counts()) > 5]    
            df = df[df['emp_loc'].map(df['emp_loc'].value_counts()) > 3]


        #df['raw_salary'] = binning(df['raw_salary'], 10)

        TRANSFORMS = [(value, [LabelBinarizer()]) for value in ['title', 'company_loc', 'emp_loc']]
        TRANSFORMS += [(['exp'], [OrdinalEncoder()])]
        #TRANSFORMS += [(['raw_salary'], [StandardScaler()])]
        TRANSFORMS += [(['raw_salary'], [SimpleImputer()])]

        self.salary_sd = np.std(np.array(df['raw_salary']))
        self.salary_avg = np.average(np.array(df[['raw_salary']]))

        df['raw_salary'] = df['raw_salary'] - self.salary_avg
        df['raw_salary'] = df['raw_salary']/(self.salary_sd)

        TRANSFORMS += [(['year'], OrdinalEncoder())]

        mapper = DataFrameMapper(TRANSFORMS, df_out=True)
        return mapper.fit_transform(df)