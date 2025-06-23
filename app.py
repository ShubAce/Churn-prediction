import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd
import pickle

model = load_model("model.h5")

with open("Label_encoder_gender.pkl",'rb') as file:
    label_encoder_gender = pickle.load(file)

with open("onehot_encoder_geo.pkl",'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open("scaler.pkl",'rb') as file:
    scaler = pickle.load(file)

st.title("Customer Churn Predcition")

geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_)
age = st.slider("Age",18,100)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score")
estimated_salary = st.number_input("Estimated salary")
tenure = st.slider("tenure",0,10)
num_of_products = st.slider("number of products", 1,4)
has_cr_card = st.selectbox("has credit card",[0,1])
is_active_member = st.selectbox("is active member",[0,1])

input_data = {
    "CreditScore":[credit_score],
    "Geography":[geography],
    "Gender":[gender],
    "Age":[age],
    "Tenure":[tenure],
    "Balance":[balance],
    "NumOfProducts":[num_of_products],
    "HasCrCard":[has_cr_card],
    "IsActiveMember":[is_active_member],
    "EstimatedSalary":[estimated_salary]
}

geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded,columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

input_df = pd.DataFrame(input_data)
input_df['Gender'] = input_df['Gender'].apply(lambda x: x[0] if isinstance(x, list) else x)
input_df['Gender'] = label_encoder_gender.transform(input_df['Gender'])
input_df = pd.concat([input_df.drop("Geography", axis=1), geo_encoded_df], axis=1)

input_data_Scaled = scaler.transform(input_df)

prediction = model.predict(input_data_Scaled)
prediction_proba= prediction[0][0]

if prediction_proba > 0.5:
    st.write("Customer is likely to churn")
else:
    st.write("customer not likely to churn")