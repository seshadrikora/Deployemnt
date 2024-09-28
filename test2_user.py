import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LogisticRegression
from pickle import dump
from  pickle import load
import seaborn as sns
import joblib

st.title('Deployment in local system')
# df=sns.load_dataset('iris')
# x=df.iloc[:,:4]
# y=df.iloc[:,4]

encoder=joblib.load('encoder.pkl')
scaler=joblib.load('scaler.pkl')
model=joblib.load('model.pkl')

st.subheader('Enter the data')

x=st.number_input('Enter SL',value=0.0)
y=st.number_input('Enter SW',value=0.0)
z=st.number_input('Enter PL',value=0.0)
a=st.number_input('Enter PW', value=0.0)

if st.button('Predict'):
    arr1=np.array([[x,y,z,a]])
    scaled_feautures=scaler.transform(arr1)
    prediction=model.predict(scaled_feautures)
    # st.subheader('actual class')
    final_predict=encoder.inverse_transform(list(prediction))
    st.subheader('Predicted class')
    st.write(final_predict)