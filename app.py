import streamlit as st
import pickle
import pandas as pd

# Load the dataset (for mapping and dropdowns)
df = pd.read_csv("new_global_air_pollution_data.csv")

# Build mapping from numeric → category
category_map = dict(zip(df["pm2_5_aqi_numeric"], df["pm2.5_aqi_category"]))

# Load the trained model
model = pickle.load(open("global_air_pollution_data_Model.pkl", "rb"))

st.title("🌍 Air Status Prediction App")
st.write("This app predicts the *Air Status (PM2.5 AQI Category)* based on global pollution indicators.")

# --- Country dropdown (show names, return codes) ---
country_options = df[['country_code', 'country_name']].drop_duplicates()
country_name = st.selectbox("Select Country", country_options['country_name'].tolist())
country_code = int(country_options[country_options['country_name'] == country_name]['country_code'].iloc[0])

# --- City dropdown (show names, return codes) ---
city_options = df[['city_code', 'city_name']].drop_duplicates()
city_name = st.selectbox("Select City", city_options['city_name'].tolist())
city_code = int(city_options[city_options['city_name'] == city_name]['city_code'].iloc[0])

# Numeric inputs
aqi_value = st.number_input("AQI Value", min_value=0.0, step=1.0)
aqi_numeric = st.number_input("AQI Numeric", min_value=0, step=1)
co_aqi_value = st.number_input("CO AQI Value", min_value=0.0, step=1.0)
co_aqi_numeric = st.number_input("CO AQI Numeric", min_value=0, step=1)
ozone_aqi_value = st.number_input("Ozone AQI Value", min_value=0.0, step=1.0)
ozone_aqi_numeric = st.number_input("Ozone AQI Numeric", min_value=0, step=1)
no2_aqi_value = st.number_input("NO₂ AQI Value", min_value=0.0, step=1.0)
no2_aqi_numeric = st.number_input("NO₂ AQI Numeric", min_value=0, step=1)
pm25_aqi_value = st.number_input("PM2.5 AQI Value", min_value=0.0, step=1.0)

# Prediction
if st.button("Predict"):
    features = [[
        country_code, city_code, aqi_value, aqi_numeric,
        co_aqi_value, co_aqi_numeric,
        ozone_aqi_value, ozone_aqi_numeric,
        no2_aqi_value, no2_aqi_numeric,
        pm25_aqi_value
    ]]
    
    # Predict numeric value (regression output)
    raw_prediction = model.predict(features)[0]
    
    # Round to nearest integer
    prediction = int(round(raw_prediction))
    
    # Map numeric → category label
    pred_label = category_map.get(prediction, "Unknown")
    
    st.success(f"🌟 Predicted Air Status: {pred_label}")
    st.text("Thank you for using the application")