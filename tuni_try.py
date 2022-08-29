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
import plotly.figure_factory as ff
import altair as alt
import base64
import csv

#@st.cache
st.header("Fruad Detection in Tunisia")
# Import the train data
train = pd.read_csv('SUPCOM_Train.csv')

# Import the test data
test = pd.read_csv('SUPCOM_Test.csv')

Submission = pd.read_csv('SUPCOM_SampleSubmission.csv')

st.subheader("Convert the dummy variables to Categorical values")
train3 = pd.get_dummies(train.drop(['id'], axis = 1))
st.write(train3.shape, "Row,  column" )
#st.write()
# Solving the dummy variable trap
train3.drop(['CTR_CATEGO_X_N'], axis = 1, inplace = True)
st.write(train3.head())
st.subheader("Data Description")
st.write(train3.describe())
train3.head()
#Visualisation

#def plot_all_scatter(dataframe, scatter_columns):
    
from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()
data_sc_raw = scaler.fit_transform(train3)
data_sc_raw = pd.DataFrame(data_sc_raw, columns = train3.columns)
traiin3 = data_sc_raw
st.write(train3.columns)
data_sc_raw.head()
data = data_sc_raw
data.head()

def user_input_features():
    BCT_CODBUR = st.sidebar.slider('BCT_CODBUR', 4.3, 7.9, 5.4)
    CTR_MATFIS = st.sidebar.slider('CTR_MATFIS_lenght', 2.0, 4.4, 3.4)
    FJU_CODFJU = st.sidebar.slider('FJU_CODFJU_lenght', 1.0, 6.9, 1.3)
    CTR_CESSAT = st.sidebar.slider('CTR_CESSAT_width', 0.1, 2.5, 0.2)
    ACT_CODACT = st.sidebar.slider('ACT_CODACT_width', 0.1, 2.5, 0.2)
    CTR_OBLACP = st.sidebar.slider('CTR_OBLACP_width', 0.1, 2.5, 0.2)
    CTR_OBLRES = st.sidebar.slider('CTR_OBLRES_width', 0.1, 2.5, 0.2)
    CTR_OBLFOP = st.sidebar.slider('CTR_OBLFOP_width', 0.1, 2.5, 0.2)
    CTR_OBLTFP = st.sidebar.slider('CTR_OBLTFP_width', 0.1, 2.5, 0.2)
    FAC_MFODEC_F = st.sidebar.slider('FAC_MFODEC_F_width', 0.1, 2.5, 0.2)
    FAC_MNTDCO_F = st.sidebar.slider('FAC_MNTDCO_F_width', 0.1, 2.5, 0.2)
    FAC_MNTTVA_F = st.sidebar.slider('FAC_MNTTVA_F_width', 0.1, 2.5, 0.2)
    FAC_MNTPRI_C = st.sidebar.slider('FAC_MNTPRI_C_width', 0.1, 2.5, 0.2)
    FAC_MFODEC_C = st.sidebar.slider('FAC_MFODEC_C_width', 0.1, 2.5, 0.2)
    FAC_MNTDCO_C = st.sidebar.slider('CTR_CESSAT_width', 0.1, 2.5, 0.2)
    FAC_MNTTVA_C = st.sidebar.slider('FAC_MNTTVA_C_width', 0.1, 2.5, 0.2)
    CTR_CATEGO_X_C = st.sidebar.slider('CTR_CATEGO_X_C_width', 0.1, 2.5, 0.2)
    CTR_CATEGO_X_M = st.sidebar.slider('CTR_CATEGO_X_M_width', 0.1, 2.5, 0.2)
    CTR_CATEGO_X_P = st.sidebar.slider('CTR_CATEGO_X_P_width', 0.1, 2.5, 0.2)
    data = {'sepal_lenght': sepal_lenght,
            'sepal_width': sepal_width,
            'petal_lenght': petal_lenght,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()
st.subheader("User Input Parameter")
st.write(df)

from sklearn.model_selection import train_test_split
X = data.drop(['target'], axis = 1)
y = data.target.values

#st.plotly_chart(data)
#st.write(plt.plot(data))
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
data.columns
# Visualisation
st.subheader("VISUALISATION")
columns_data = ['BCT_CODBUR', 'CTR_MATFIS', 'FJU_CODFJU', 'CTR_CESSAT', 'ACT_CODACT',
       'CTR_OBLDIR', 'CTR_OBLACP', 'CTR_OBLRES', 'CTR_OBLFOP', 'CTR_OBLTFP',
       'FAC_MFODEC_F', 'FAC_MNTDCO_F', 'FAC_MNTTVA_F', 'FAC_MNTPRI_C',
       'FAC_MFODEC_C', 'FAC_MNTDCO_C', 'FAC_MNTTVA_C', 'CTR_CATEGO_X_C',
       'CTR_CATEGO_X_M', 'CTR_CATEGO_X_P']
#fig = ff.create_distplot(
#         data,columns_data, bin_size=[.1, .25, .5])
#st.plotly_chart(data, use_container_width=True)
#st.altair_chart()

st.subheader("1. Modeling the data with Catboost")
# Model 1 - CatBoost
from catboost import CatBoostRegressor
regressor = CatBoostRegressor()
regressor.fit(x_train, y_train)

prediction = regressor.predict(x_test)
#prediction = pd.DataFrame(prediction, columns='target')
prediction
st.write(mean_squared_error(y_test, prediction))
#p = alt.Chart(prediction, y_test).scatter()
prediction_df = pd.DataFrame(prediction, columns=['Predict'])
#alt.BoxPlot(data)
#st.sns_heatmap(data)
#st.write(sns.heatmap(data))
#st.altair_chart(data)

def filesdownload(predictiion):
    csv = prediction_df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode() #string <-> convention
    href = f'<a href="data:file/csv;base64,{b64}" download="prediction.csv">Download Predicted CSV File</a>'
    return href
st.markdown(filesdownload(prediction_df), unsafe_allow_html=True)
st.button('Good work')
st.header("Visualisation")