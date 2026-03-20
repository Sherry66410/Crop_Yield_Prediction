import streamlit as st
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb

# --- 1. Load the trained model and preprocessors ---

# Load the XGBoost model
with open('xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the StandardScaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load the LabelEncoder
with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# Load feature names
with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

# Define the original categorical columns for one-hot encoding
original_categorical_cols = ['Soil_Type', 'Crop', 'Weather_Condition']

# --- 2. Preprocessing Function ---
def preprocess_input(input_df: pd.DataFrame) -> pd.DataFrame:
    # Apply StandardScaler for Rainfall_mm and Temperature_Celsius
    input_df[['Rainfall_mm', 'Temperature_Celsius']] = scaler.transform(input_df[['Rainfall_mm', 'Temperature_Celsius']])

    # Apply LabelEncoder for Fertilizer_Used and Irrigation_Used
    input_df['Fertilizer_Used'] = le.transform(input_df['Fertilizer_Used'])
    input_df['Irrigation_Used'] = le.transform(input_df['Irrigation_Used'])

    # Apply OneHotEncoder for categorical features
    input_df_processed = pd.get_dummies(input_df, columns=original_categorical_cols)

    # Align columns with the training data features (crucial step)
    # Create a DataFrame with all expected features, filled with zeros
    final_input = pd.DataFrame(0, index=[0], columns=feature_names)

    # Copy values from the processed input to the final_input
    for col in input_df_processed.columns:
        if col in final_input.columns:
            final_input[col] = input_df_processed[col]

    # Ensure boolean columns from get_dummies are int
    bool_cols_final = final_input.select_dtypes(include='bool').columns
    final_input[bool_cols_final] = final_input[bool_cols_final].astype(int)

    return final_input

# --- 3. Streamlit App Layout ---
st.set_page_config(
    page_title="Crop Yield Prediction App", 
    layout="centered", 
    initial_sidebar_state="expanded", 
    page_icon="🌾"
)

st.title("🌾 Crop Yield Prediction App")
st.write("Enter the crop growing conditions to predict the yield (tons per hectare).")

# Custom background
def set_custom_background(color):
    st.markdown(
        f"""
        <style>
        .stApp {{background-color: {color};}}
        </style>
        """,
        unsafe_allow_html=True
    )

set_custom_background("#000000")  # Black background

# Input fields
st.header("Input Crop Conditions")

soil_type = st.selectbox(
    "🌍 Soil Type", 
    ['Sandy', 'Clay', 'Loam', 'Peaty', 'Chalky', 'Silt']
)

crop_type = st.selectbox(
    "🌱 Crop Type", 
    ['Cotton', 'Rice', 'Barley', 'Soybean', 'Wheat', 'Maize']
)

rainfall = st.number_input(
    "🌧️ Rainfall (mm)", 
    min_value=100.0, 
    max_value=999.0, 
    value=500.0, 
    step=10.0
)

temperature = st.number_input(
    "☀️ Temperature (Celsius)", 
    min_value=15.0, 
    max_value=40.0, 
    value=25.0, 
    step=0.5
)

fertilizer_used = st.checkbox("🧪 Fertilizer Used?")
irrigation_used = st.checkbox("🚿 Irrigation Used?")

weather_condition = st.selectbox(
    "☁️ Weather Condition", 
    ['Cloudy', 'Rainy', 'Sunny']
)

days_to_harvest = st.number_input(
    "⏳ Days to Harvest", 
    min_value=60, 
    max_value=149, 
    value=100, 
    step=1
)

# --- 4. Make Prediction ---
if st.button("🔮 Predict Yield", type="primary"):
    input_data = pd.DataFrame({
        'Soil_Type': [soil_type],
        'Crop': [crop_type],
        'Rainfall_mm': [rainfall],
        'Temperature_Celsius': [temperature],
        'Fertilizer_Used': [fertilizer_used],
        'Irrigation_Used': [irrigation_used],
        'Weather_Condition': [weather_condition],
        'Days_to_Harvest': [days_to_harvest]
    })

    processed_input = preprocess_input(input_data)

    try:
        prediction = model.predict(processed_input)
        st.success(f"🎯 Predicted Crop Yield: **{prediction[0]:.2f} tons per hectare**")
        
        # Additional visualization
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Yield", f"{prediction[0]:.2f} t/ha")
        with col2:
            st.metric("Rainfall", f"{rainfall} mm")
        with col3:
            st.metric("Temperature", f"{temperature}°C")
            
    except Exception as e:
        st.error(f"❌ An error occurred during prediction: {e}")

st.markdown("---")
st.write("This app uses an XGBoost Regressor model to predict crop yield based on various environmental and agricultural factors.")
st.caption("Made with ❤️ using Streamlit")
