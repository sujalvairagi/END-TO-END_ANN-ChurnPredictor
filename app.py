import streamlit as st
import numpy as np 
import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import load_model

#Load the trained model
#LOAD THE TRAINED MODEL )
model =load_model('model.keras')
#LOAD THE SCALER
with open('scaler.pkl', 'rb') as file:
    scaler= pickle.load(file)
#LOAD THE ENCODER
with open('onehot_encoder.pkl', 'rb') as file:
    label_encoder_geo = pickle.load(file)
#LOAD THE COLUMNS
with open('label_encoder.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

#streamlit app
st.title("Customer Churn Prediction")

#user inputs
#Get user input
geography = st.selectbox('Geography', label_encoder_geo.categories_[0])  
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18,92)
balance = st.number_input('Balance')

estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0,10)
num_of_products = st.slider('Number of Products', 1,4)
# Has Credit Card: show Yes/No to user, map to 0/1
has_cr_card_choice = st.selectbox('Has Credit Card', ['Yes', 'No'])
has_cr_card = 1 if has_cr_card_choice == 'Yes' else 0

# Is Active Member: show Yes/No to user, map to 0/1
is_active_member_choice = st.selectbox('Is Active Member', ['Yes', 'No'])
is_active_member = 1 if is_active_member_choice == 'Yes' else 0
credit_score = st.slider('Credit Score', 350,850)
#Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender':[label_encoder_gender.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_member],
    'EstimatedSalary':[estimated_salary],

})

geo_encoded = label_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=label_encoder_geo.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)


#scale the input data  
input_data_scaled = scaler.transform(input_data)
#Make prediction
prediction = model.predict(input_data_scaled)
churn_probability = prediction[0][0]

if prediction >= 0.5:
   st.write('Customer is likely to churn')
else:
    st.write('Customer is unlikely to churn')