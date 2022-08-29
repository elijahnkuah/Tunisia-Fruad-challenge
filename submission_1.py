#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
import re
import timeit
import random
random.seed(3)

# In[2]:


# WORK ON OUTLINES USING LOG
# 
# Import the train data
train = pd.read_csv('datasets/SUPCOM_Train.csv')

# Import the test data
test = pd.read_csv('datasets/SUPCOM_Test.csv')

Submission = pd.read_csv('datasets/SUPCOM_SampleSubmission.csv')

# In[3]:


target = train.target.values
print(target.shape)

# Combine train and test data for easy preprocessing
ntrain = train.shape[0]
ntest = test.shape[0]
data = pd.concat((train, test), sort=False).reset_index(drop = True)
print(data.shape)
data.head()
data.info()

## Data quality check

# In[7]:


# This function seperate the columns into the various data types
def data_types(dataframe):
  int_col = []
  continuous_col = []
  cat_col = []
  for column in train.columns:
    if train[column].dtypes == 'int64':
      int_col.append(column)
    elif train[column].dtypes == 'float':
      continuous_col.append(column)
    else:
      cat_col.append(column)
  return int_col, continuous_col, cat_col


# In[8]:


int_col, continuous_col, cat_col = data_types(data)


# In[9]:


print('Ordinal Variables', len(int_col))
print('Continuous Variables',len(continuous_col))
print('Categorical Variables', len(cat_col))


# ### Understanding the int64 data type variables

# In[10]:


data[int_col].describe()


# In[11]:


data.CTR_CATEGO_X.unique()


# In[12]:


data.head()


# In[13]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
LE = LabelEncoder()


# In[14]:


#data[8] = LE.fit_transform(data[8])


# In[15]:


# All the values in some variables are only zeros. 
col_only_zeros = []
for column in continuous_col:
  if data[column].max() == 0 and data[column].min() == 0:
    col_only_zeros.append(column)

print(len(col_only_zeros))
print(col_only_zeros)


# In[16]:


print(cat_col)


# In[17]:


int_col_use = ['CTR_OBLDIR', 'CTR_OBLACP', 'CTR_OBLRES', 'CTR_OBLFOP', 'CTR_OBLTFP', 'CTR_OBLDCO', 'CTR_OBLTVA', 'CTR_OBLTCL', 'CTR_RATISS']
int_col_drop = ['BCT_CODBUR', 'CTR_MATFIS', 'FJU_CODFJU', 'CTR_CESSAT', 'ACT_CODACT', 'EXE_EXERCI', 'RES_ANNIMP']


# In[18]:


redundant_col = int_col_drop + col_only_zeros + ['id']
print(len(redundant_col),redundant_col)


# In[19]:


# Function to handle missing values
def missing_values(data, threshold = 70.0):
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


# In[20]:


# Columns to drop and those to keep due to missing values
drop_columns , keep_columns = missing_values(data)

print('Number of columns with more than 70% missing values:', len(drop_columns))
print('Number of columns to keep:', len(keep_columns))


# In[21]:


# Columns to keep 
data[keep_columns].head()


# In[22]:


# Drop redundant columns
data2 = data.drop(redundant_col, axis = 1)
print(data2.shape)
data2.head()


# In[23]:


data2.CTR_CATEGO_X.unique()


# In[24]:


data2.head()


# In[25]:


# Convert the Categorical columns to dummies
data_dumi = pd.get_dummies(data2)


# In[26]:


data_dumi.tail()


# In[27]:


data.drop('id', inplace=True, axis=1)


# In[28]:


data.head()


# In[29]:


data.tail()


# In[30]:


data = pd.get_dummies(data)


# In[31]:


data.head()


# In[32]:


data.shape


# # SCALING THE DATA 

# In[33]:


from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()
data_sc_raw = scaler.fit_transform(data)
data_sc_raw = pd.DataFrame(data_sc_raw, columns = data.columns)


# In[34]:


data_sc_raw.head(4)


# In[35]:


#Train and test datasets
train = data_sc_raw[:ntrain].copy()
test = data_sc_raw[ntrain:].copy()
test = test.reset_index(drop=True)


# In[36]:


print('train:',train.shape)
print('test:', test.shape)

target.shape
# In[37]:


# lightgbm for regression
from numpy import mean
from numpy import std
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from matplotlib import pyplot

# evaluate the model
model = LGBMRegressor()
cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, train, target, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1, error_score='raise')
print('LGBM MSE: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
# fit the model on the whole dataset
model_lg = LGBMRegressor()
model_lg.fit(train, target)


# In[38]:


# catboost for regression
from numpy import mean
from numpy import std
from catboost import CatBoostRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from matplotlib import pyplot
# evaluate the model
model = CatBoostRegressor(verbose=0, n_estimators=100)
cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, train, target, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1, error_score='raise')
print('CAT MSE: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
# fit the model on the whole dataset
model_cat = CatBoostRegressor(verbose=0, n_estimators=100)
model_cat.fit(train, target)


# In[ ]:


# xgboost for regression
from numpy import mean
from numpy import std
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from matplotlib import pyplot

# evaluate the model
model = XGBRegressor(objective='reg:squarederror')
cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, train, target, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1, error_score='raise')
print('XGBM MSE: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
# fit the model on the whole dataset
model_xg = XGBRegressor(objective='reg:squarederror')
model_xg.fit(train, target)


# In[40]:


#pip install mlxtend  


# In[ ]:


from sklearn.linear_model import RidgeCV
from sklearn.svm import LinearSVR
#from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor


# In[ ]:





# In[46]:


# make a prediction with a stacking mlxtend
from sklearn.linear_model import LinearRegression
#import mlxtend
#from sklearn.ensemble import StackingRegressor
from mlxtend.regressor import StackingRegressor

# define meta learner model
lin_reg = LinearRegression()
# define the stacking ensemble
model = StackingRegressor(regressors=[model_lg, model_cat, model_xg], meta_regressor=lin_reg)
# fit the model on all available data
model.fit(train, target)
stack_result = model.predict(test)


# In[ ]:


#conda install mlxtend


# # SUBMISSION

# In[47]:


Submission['target'] = stack_result #model_lg.predict(test) 
file_name = 'submission12-stack3-raw.csv'
Submission.to_csv(file_name ,index=False)


# In[ ]:




