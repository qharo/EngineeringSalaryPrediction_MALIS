# Data Science Salary Prediction

This is a machine learning model to help predict the average salaries of data science positions in several different countries. (Note that the name of the repository, "Engineering Salary Prediction," is innacurate as the model was originally scoped to predict engineering salaries but was later rescoped to predict data science salaries.) The dataset used for this model is located at the following directory: /data/archive/ds_salaries.csv. The model includes the following experiments, each utilizing a different machine learning method and programmed in its own seperate file:

The seventh and final experiment was performed using the ensemble methodology. As mentioned previously, our ensemble experiment incorporated the following methods: 

exp4.py: Utilizes the SVM method; first with linear, then RBF, and then the polynomial kernel. For the polynomial kernel, the degrees that were configured and run with were 2, 7, and then 11.

exp5.py: Utilizes the following regression methods with default configurations: linear, lasso, and ridge.

exp6.py: Utilizes RFC. The RFC method was configured and run with 100, 250, 500, and then 750 trees.

exp7.py: Utilizes the KNN method. The nearest neighbors estimator of the KNN method was configured and run with 1, 5, 7, 9, 10, 11, 30, 60, 90, and then 100 nearest neighbors.

exp8.py: Utilizes tuned versions of the lasso and ridge methods. The ridge regression method was run each time with alpha being tuned between 0.0 and 0.9, stepping up by 0.1 for each subsequent run. The lasso regression method was run each time with the number of iterations being tuned between 1 and 901, stepping up by 100 for each subsequent run.

exp9.py: Utilizes a neural network that was configured to run in 10 epochs.

exp10.py: Utilizes an ensemble that incorporates the following methods linear, lasso, and ridge regression, KNN, and RFC. For each incorporated method, their hyperparameters were tuned as follows: lasso regression with 100 iterations, ridge regression with an alpha of 0.9, KNN with an estimator of 7 nearest neighbors, and RFC with 250 forests. 

To run all of the experiments, cd into the root directory of the repository and input the following command in a terminal: python3 prediction.py.
