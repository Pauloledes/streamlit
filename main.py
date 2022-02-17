import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()

with header:
    st.title('Welcome to my awesome data science project!')
    st.text('In this project I look into the transactions of taxis in NYC.')


with dataset:
    st.header('NYC taxi dataset')
    st.text('I found this dataset on https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page')

    taxi_data = pd.read_csv('data/taxi_data_r.csv')
    st.write(taxi_data.head())

    distribution_pickup = pd.DataFrame(taxi_data['PULocationID'].value_counts())
    st.bar_chart(distribution_pickup)


with features:
    st.header('The features I created')

    st.markdown('* **first feature:** I created this feature because of this... I calculated it using...')
    st.markdown('* **second feature:** I created this feature because of this... I calculated it using...')


with model_training:
    st.header('Time to train the model !')
    st.text('Here you get to choose the hyperparameters of the model')

    sel_col, disp_col = st.columns(2)

    max_depth = sel_col.slider('What should be the max_depth of the model?', min_value=10, max_value=100,
                   value=20, step=10)

    n_estimators = sel_col.selectbox('How many tress should there be?', options=[100, 200, 300, 'No limit'], index=0)

    if n_estimators == 'No limit':
        n_estimators = 1000

    input_feature = sel_col.selectbox('Which feature should be used as the input feature ?', options=taxi_data.columns, index=taxi_data.columns.get_loc('PULocationID'))

    regr = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)

    X = taxi_data[[input_feature]]
    y = taxi_data[['trip_distance']]
    y = np.ravel(y)

    regr.fit(X, y)
    prediction = regr.predict(X)

    disp_col.subheader('Mean absolute error of the model is :')
    disp_col.write(mean_absolute_error(y, prediction))

    disp_col.subheader('Mean squared error of the model is')
    disp_col.write(mean_squared_error(y, prediction))

    disp_col.subheader('R2 score of the model is')
    disp_col.write(r2_score(y, prediction))

