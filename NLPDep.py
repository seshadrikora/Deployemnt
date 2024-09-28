import pandas as pd
import numpy as np
import streamlit as st
import joblib
import re
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from nltk.stem import PorterStemmer



st.title('Deployemnt for review')

vect=joblib.load('tfidf.pkl')
model=joblib.load('modelnlp.pkl')
print(type(model))
stemmer=PorterStemmer()
st.subheader('Please enter your review')

text=st.text_input('Pleae enter your review')                   

def preprocessing(text):
    text=text.lower()
    text=re.sub('[^a-zA-Z]','',text)
    text=text.split()
    text=[stemmer.stem(word) for word in text if word not in stopwords.words('english')]
    text=' '.join(text)
    return text

if st.button('Predict'):

    # Step 1: Preprocess the input text
    text = preprocessing(text)

    # Step 2: Vectorize the preprocessed text (fix transform argument to a list)
    vector = vect.transform([text]).toarray()

    # Step 3: Make prediction using the trained model
    pred = model.predict(vector)

    # Step 4: Display the predicted class
    st.subheader('Predicted Class')
    st.write(f"Model Prediction: {pred[0]}")
    # Step 5: Check the prediction and output the result accordingly
    if pred[0] == 0:
        st.write('Negative')  # Assuming 0 is for negative reviews
    else:
        st.write('Positive')  # Assuming 1 is for positive reviews

    
   
        

