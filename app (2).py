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

# Custom CSS for better styling
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .main-title {
        text-align: center;
        color: white;
        font-size: 3em;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        margin-bottom: 10px;
    }
    
    .subtitle {
        text-align: center;
        color: white;
        font-size: 1.2em;
        margin-bottom: 30px;
    }
    
    .yield-display {
        background: white;
        padding: 40px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        margin: 20px 0;
    }
    
    .yield-value {
        font-size: 5em;
        font-weight: bold;
        color: #667eea;
        margin: 10px 0;
    }
    
    .yield-unit {
        font-size: 1.5em;
        color: #666;
        margin-top: -10px;
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 2em;
        font-weight: bold;
    }
    
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 1.3em;
        font-weight: bold;
        padding: 15px;
        border-radius: 10px;
        border: none;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
    }
    
    section[data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.95);
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-title">🌾 Crop Yield Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Predict your crop yield based on environmental conditions</p>', unsafe_allow_html=True)

# Sidebar for inputs
with st.sidebar:
    st.header("🔧 Input Parameters")
    
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
        "🌡️ Temperature (°C)", 
        min_value=15.0, 
        max_value=40.0, 
        value=25.0, 
        step=0.5
    )

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
    
    st.markdown("---")
    
    fertilizer_used = st.checkbox("🧪 Fertilizer Used", value=False)
    irrigation_used = st.checkbox("🚿 Irrigation Used", value=False)
    
    st.markdown("---")
    
    predict_button = st.button("🔮 Predict Yield", type="primary", use_container_width=True)

# Main content area
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    # --- 4. Make Prediction ---
    if predict_button:
        with st.spinner('🔄 Calculating yield prediction...'):
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
                
                # Clamp yield to range 1.0 to 6.5 tons per hectare
                yield_value = np.clip(prediction[0], 1.0, 6.5)
                yield_value = round(yield_value, 1)  # Round to 1 decimal place
                
                # Display result in a beautiful card
                st.markdown(f"""
                    <div class="yield-display">
                        <h2 style="color: #333; margin-bottom: 10px;">🎯 Predicted Crop Yield</h2>
                        <div class="yield-value">{yield_value}</div>
                        <div class="yield-unit">tons per hectare</div>
                    </div>
                """, unsafe_allow_html=True)
                
                st.success(f"✅ Prediction completed successfully!")
                    
            except Exception as e:
                st.error(f"❌ An error occurred during prediction: {e}")
    else:
        # Welcome message when no prediction has been made
        st.info("👈 Enter your crop parameters in the sidebar and click **Predict Yield** to get started!")
        
        st.markdown("### 📋 How to Use")
        st.markdown("""
        1. **Select** your soil type and crop variety
        2. **Enter** environmental conditions (rainfall, temperature, weather)
        3. **Specify** days to harvest
        4. **Check** if using fertilizer and/or irrigation
        5. **Click** the Predict Yield button
        
        The model will calculate the expected yield in **tons per hectare** (range: 1.0 - 6.5).
        """)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: white;'>
        <p>This app uses an <b>XGBoost Regressor</b> model trained on agricultural data.</p>
        <p>Yield predictions range from <b>1.0 to 6.5 tons per hectare</b>.</p>
        <p style='margin-top: 20px;'>Made with ❤️ using Streamlit</p>
    </div>
""", unsafe_allow_html=True)
