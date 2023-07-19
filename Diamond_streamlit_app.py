#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the model from the pickle file
model = pickle.load(open('diamond_pred.pkl', 'rb'))

# Define categorical columns for one-hot encoding
categorical_cols = ['cut', 'color', 'clarity']

# Define numerical columns for scaling
numerical_cols = ['carat', 'depth', 'table', 'x', 'y', 'z']

# Create a DataFrame with default values for user input
default_data = {
    'carat': 0.5,
    'depth': 60.0,
    'table': 55.0,
    'cut': 'Ideal',
    'color': 'D',
    'clarity': 'IF',
    'x': 5.0,
    'y': 5.0,
    'z': 3.0
}
input_df = pd.DataFrame(default_data, index=[0])

#



# Streamlit app
st.title("Diamond Price Prediction")
st.write("Enter the diamond features below to predict the price:")

# Create input fields for each feature
carat = st.number_input("Carat", min_value=0.0, value=0.5)
depth = st.number_input("Depth", min_value=0.0, value=60.0)
table = st.number_input("Table", min_value=0.0, value=55.0)
cut = st.selectbox("Cut", options=['Ideal', 'Premium', 'Very Good', 'Good', 'Fair'])
color = st.selectbox("Color", options=['D', 'E', 'F', 'G', 'H', 'I', 'J'])
clarity = st.selectbox("Clarity", options=['IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2', 'I1'])
x = st.number_input("x", min_value=0.0, value=5.0)
y = st.number_input("y", min_value=0.0, value=5.0)
z = st.number_input("z", min_value=0.0, value=3.0)

# Update the input DataFrame with user values
input_df = pd.DataFrame({
    'carat': carat,
    'depth': depth,
    'table': table,
    'cut': cut,
    'color': color,
    'clarity': clarity,
    'x': x,
    'y': y,
    'z': z
}, index=[0])


# Predict the price based on user input
predicted_price = model.predict(input_df)

st.write(f"Predicted Price: {predicted_price}")


# In[ ]:




