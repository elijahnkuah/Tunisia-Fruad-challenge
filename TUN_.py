import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import catboost as cat_
import seaborn as sns
import lightgbm as lgb

# WORK ON OUTLINES USING LOG
# 
# Import the train data
train = pd.read_csv('SUPCOM_Train.csv')

# Import the test data
test = pd.read_csv('SUPCOM_Test.csv')

Submission = pd.read_csv('SUPCOM_SampleSubmission.csv')

# Function to handle missing values train data
def missing_values(data, threshold = 40.0):
    global drop_columns
    global keep_columns
    global column
    drop_columns = []
    keep_columns = []
    columns = data.columns 
    for column in columns:
        missing_val_percent = (100 * data[column].isna().sum() / len(data))
        if missing_val_percent >= threshold:
            drop_columns.append(column)
        else:
            keep_columns.append(column)
    return (drop_columns, keep_columns)

'''# Function to handle missing values in test data
def missing_values_test(data_test, threshold=40.0):
    drop_columns_test = []
    keep_columns_test = []
    columns = data_test.columns
    for column in columns:
        missing_val_test_percent = (100 *data[column].isna().sum()/len(data_test))
        if missing_val_test_percent >= threshold:
            drop_columns_test.append(column)
        else:
            keep_columns_test.append(column)
    return(drop_columns_test, keep_columns_test)'''
    
'''
def missing_values(data, threshold = 40.0):
    drop_columns = []
    keep_columns = []
    columns = data.columns 
    for column in columns:
        missing_val_percent = (100 * data[column].isna().sum() / len(data))
        if missing_val_percent >= threshold:
            drop_columns.append([column, missing_val_percent])
        else:
            keep_columns.append([column, missing_val_percent])
    return (drop_columns, keep_columns)
'''

# Droping columns with missing data above the threshold
drop_columns, keep_columns = missing_values(train, 40)
train2 = train.drop(drop_columns, axis = 1)
train2.head()

# Convert the Categorical columns to dummies
train3 = pd.get_dummies(train2.drop(['id'], axis = 1))

# Solving the dummy variable trap
train3.drop(['CTR_CATEGO_X_N'], axis = 1, inplace = True)
train3.head()

# COONECTING THE TWO DATA SETS
# CHECK THIS FUNCTION
def missing_columns_test(data1):
    global drop_columns
    global keep_columns
    global drop
    drop_columns_test = []
    keep_columns_test = []
    columns_1 = data1.columns
    for drop in drop_columns:
        if drop == column1 :
            drop_columns_test.append(column1)
        else:
            keep_columns_test.append(column1)
    return(drop_colums_test, keep_columns_test)
    
    
drop_columns_test, keep_columns_test = missing_columns_test(test)
test2 = test.drop(drop_columns_test, axis = 1)
test3 = pd.get_dummies(test2.drop(['id'], axis = 1))
test3.drop(['CTR_CATEGO_X_N'], axis = 1, inplace = True)

from sklearn.model_selection import train_test_split
x_train = train3.drop(['target'], axis = 1)
y_train = train3.target.values

#x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Model 1 - CatBoost
from catboost import CatBoostRegressor
regressor = CatBoostRegressor()
regressor.fit(x_train, y_train)

prediction = regressor.predict(test3)

# Evaluate
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, prediction)

# Model 2 - Light gbm
from lightgbm import LGBMRegressor
regressor = LGBMRegressor()
regressor.fit(x_train, y_train)

prediction_2 = regressor.predict(x_test)
mean_squared_error(y_test, prediction_2)

# Model 3 - Xgboost
from xgboost.sklearn import XGBRegressor
regressor = XGBRegressor()
regressor.fit(x_train, y_train)

prediction_3 = regressor.predict(x_test)
mean_squared_error(y_test, prediction_3)

#Exporting to excel
train3 = pd.DataFrame(train3)
train3 = train3.to_excel(r'C:\Users\Elijah Nkuah\Documents\python\datascience\Tunisia Fruad challenge\train_data.xlsx', index=False, header=True)

prediction = pd.DataFrame(prediction)
prediction = prediction.to_excel(r'C:\Users\Elijah Nkuah\Documents\python\datascience\Tunisia Fruad challenge\prediction.xlsx', index=False, header=True)

