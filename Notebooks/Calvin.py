#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# Load Data with caching and error handling
@st.cache_data
def load_data(uploaded_file=None):
    try:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_csv("climate_data_final_df.csv")  # Default file
        df.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')
        return df
    except Exception as e:
        st.error(f"Data loading failed: {str(e)}")
        return pd.DataFrame()

# Streamlit UI
st.title("🌍 Climate Change & Energy Impact Analyzer")
st.subheader("Interactive Analysis Dashboard")

# File Uploader
uploaded_file = st.file_uploader("Upload your climate dataset (CSV)", type=["csv"])

# Load data
df = load_data(uploaded_file)
if df.empty:
    st.error("No data loaded. Please upload a valid CSV file.")
    st.stop()

# Sidebar Controls
with st.sidebar:
    st.header("Analysis Controls")
    
    # Country Selection
    if 'Entity' not in df.columns:
        st.error("Dataset missing 'Entity' column for country selection")
        st.stop()
    selected_country = st.selectbox("Select Country", df['Entity'].unique())
    
    # Feature Selection
    available_features = [col for col in df.columns if col not in ['Entity', 'Year']]
    if not available_features:
        st.error("No features available in dataset")
        st.stop()
    
    # Safe default features
    possible_defaults = ['CO2 emissions (metric tons per capita)', 
                        'GDP per capita', 
                        'Population']
    safe_default_features = [feat for feat in possible_defaults if feat in available_features][:2]
    
    selected_features = st.multiselect(
        "Select Features", 
        available_features,
        default=safe_default_features
    )
    
    # Validate features
    if not selected_features:
        st.error("Please select at least one feature")
        st.stop()
    
    # Target Selection
    valid_targets = [
        'Average Temperature', 
        'mmfrom1993-2008average', 
        'Renewable energy consumption (% of total final energy consumption)'
    ]
    available_targets = [t for t in valid_targets if t in df.columns]
    
    if not available_targets:
        st.error("No valid target variables found in dataset")
        st.stop()
    
    target = st.selectbox("Select Target Variable", available_targets)

# Data Filtering
try:
    filtered_df = df[df['Entity'] == selected_country].copy()
    required_columns = ['Year'] + selected_features + [target]
    filtered_df = filtered_df[required_columns].dropna()
    
    if filtered_df.empty:
        st.error(f"No data available for {selected_country} with selected features")
        st.stop()
        
    if len(filtered_df) < 10:
        st.warning(f"Low data count ({len(filtered_df)} records). Results may be unreliable.")
        
except KeyError as e:
    st.error(f"Missing column in dataset: {str(e)}")
    st.stop()

# Data Exploration Section
st.subheader("Data Exploration")

col1, col2 = st.columns(2)
with col1:
    st.write("### Selected Data Preview")
    st.dataframe(filtered_df.head(), height=250)

with col2:
    st.write("### Target Distribution")
    fig = px.histogram(filtered_df, x=target, nbins=50)
    st.plotly_chart(fig, use_container_width=True)

# Time Series Visualization
if 'Year' in filtered_df.columns:
    st.write("### Temporal Trends")
    time_fig = px.line(filtered_df, x='Year', y=target, 
                      title=f"{target} Over Time in {selected_country}")
    st.plotly_chart(time_fig, use_container_width=True)

# Model Training Section
st.subheader("Predictive Modeling")

# Prepare data
X = filtered_df[selected_features]
y = filtered_df[target]

# Model selection
model_choice = st.radio("Select Model Type", ["Random Forest", "LSTM"], horizontal=True)

# Train/test split
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42
    )
except ValueError as e:
    st.error(f"Train/test split failed: {str(e)}")
    st.stop()

def train_model(X_train, y_train, model_type):
    try:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        if model_type == "Random Forest":
            model = RandomForestRegressor(n_estimators=200, random_state=42)
            model.fit(X_train_scaled, y_train)
            return model, scaler
            
        elif model_type == "LSTM":
            if 'Year' not in filtered_df.columns:
                st.error("LSTM requires temporal data. Ensure 'Year' column exists.")
                return None, None  # Ensures proper return even if error occurs
            
            # Reshape data for LSTM
            X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))

            model = Sequential([
                LSTM(64, return_sequences=True, input_shape=(1, X_train_scaled.shape[1])),
                Dropout(0.3),
                LSTM(32),
                Dropout(0.2),
                Dense(1)
            ])
            
            model.compile(optimizer='adam', loss='mse')
            history = model.fit(
                X_train_reshaped, y_train, 
                epochs=100, 
                batch_size=16, 
                validation_split=0.2, 
                verbose=0
            )
            
            # Plot training history
            fig = px.line(
                pd.DataFrame({
                    'Training Loss': history.history['loss'], 
                    'Validation Loss': history.history['val_loss']
                }),
                title='Model Training Progress'
            )
            st.plotly_chart(fig)
            
            return model, scaler
        
    except Exception as e:
        st.error(f"Model training failed: {str(e)}")
        return None, None  # Ensures function doesn't crash

# Call the function
model, scaler = train_model(X_train, y_train, model_choice)

if model is None or scaler is None:
    st.stop()  # Ensures script doesn't proceed with an invalid model

