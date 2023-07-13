import numpy as np
import streamlit as st
from pickle import load

model = load(open('xgb.pkl', 'rb'))

st.write("<h1 style='text-align: center; color: #FFD700;'>Medical Insurance Predictor👨‍⚕️</h1>", unsafe_allow_html = True)
st.write('This app predicts the insurance premium based on age, BMI, and smoking status.')

col1, col2 = st.columns(2)

with col1:
    st.image('image.jpeg')
with col2:
    st.image('images.jpeg')

col3, col4 = st.columns(2)

with col3:
    smoker = st.selectbox('Smoker', ['No', 'Yes'])
with col4:
    bmi = st.number_input('BMI', 12.0, 50.0, 25.0, step = 0.2)
    
age = st.slider('Age', 18, 100, 25)

smoker = 1 if smoker == 'Yes' else 0

def predict(age, bmi, smoker):
    features = np.array([age, bmi, smoker]).reshape(1, -1)
    prediction = model.predict(features)[0]
    return prediction

if st.button('Predict'):
    prediction = predict(age, bmi, smoker)
    prediction_text = f'<span style = "font-size:30px; color:#FFD700;">Predicted Insurance - ${prediction:.2f}💰</span>'
    st.write(prediction_text, unsafe_allow_html = True)
