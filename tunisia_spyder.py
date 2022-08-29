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
import streamlit as st
from sklearn.metrics import mean_squared_error

st.header("Fruad Detection in Tunisia")
# Import the train data
train = pd.read_csv('SUPCOM_Train.csv')

# Import the test data
test = pd.read_csv('SUPCOM_Test.csv')

Submission = pd.read_csv('SUPCOM_SampleSubmission.csv')

# Function to handle missing values
def missing_values(data, threshold = 40.0):
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

# Function to handle missing values

# Droping columns with missing data above the threshold
drop_columns, keep_columns = missing_values(train, 40.0)
train2 = train.drop(drop_columns, axis = 1)

st.subheader('Dropped columns with more than 40% missing variables')
st.write(train2.head())

# Convert the Categorical columns to dummies
st.subheader("Convert the Categorical columns to dummies")
train3 = pd.get_dummies(train2.drop(['id'], axis = 1))

# Solving the dummy variable trap
train3.drop(['CTR_CATEGO_X_N'], axis = 1, inplace = True)
st.write(train3.head())
st.write(train3.describe())

from sklearn.model_selection import train_test_split
X = train3.drop(['target'], axis = 1)
y = train3.target.values

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

st.subheader("1. Modeling the data with Catboost")
# Model 1 - CatBoost
from catboost import CatBoostRegressor
regressor = CatBoostRegressor()
regressor.fit(x_train, y_train)

prediction = regressor.predict(x_test)
prediction
st.write(mean_squared_error(y_test, prediction))


# Evaluate
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, prediction)

# Model 2 - Light gbm
from lightgbm import LGBMRegressor
regressor = LGBMRegressor()
regressor.fit(x_train, y_train)

prediction = regressor.predict(x_test)
mean_squared_error(y_test, prediction)

# Model 3 - Xgboost
from xgboost.sklearn import XGBRegressor
regressor = XGBRegressor()
regressor.fit(x_train, y_train)

prediction = regressor.predict(x_test)
mean_squared_error(y_test, prediction)

from sklearn.linear_model import RidgeCV
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor
#from sklearn.ensemble import StackingRegressor
#from sklearn.ensemble import StackingRegressor
from mlxtend.regressor import StackingRegressor

model = StackingRegressor(estimators=level0, final_estimator=level1, cv=5)
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import TransformerMixin
