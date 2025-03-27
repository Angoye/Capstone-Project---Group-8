#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# Configure page
st.set_page_config(page_title="Climate Analyzer", layout="wide", page_icon="🌍")

# Custom CSS for professional look
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    .st-bw {background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);}
    .metric-card {padding: 15px; background: white; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);}
    </style>
    """, unsafe_allow_html=True)

# Load Data
@st.cache_data
def load_data(uploaded_file=None):
    try:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_csv("climate_data_final_df.csv")
        df.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')
        return df
    except Exception as e:
        st.error(f"Data loading failed: {str(e)}")
        return pd.DataFrame()

# Streamlit UI
st.title("🌍 Climate Change Impact Analyzer")
st.markdown("---")

# File Upload
with st.expander("Upload Data", expanded=True):
    uploaded_file = st.file_uploader("Upload climate dataset (CSV)", type=["csv"], label_visibility="collapsed")
    df = load_data(uploaded_file)
    if df.empty:
        st.error("No data loaded. Please upload a valid CSV file.")
        st.stop()

# Sidebar Controls
with st.sidebar:
    st.header("Analysis Controls")
    selected_country = st.selectbox("Select Country", df['Entity'].unique())
    available_targets = [col for col in df.columns if col not in ['Entity', 'Year']]
    target = st.selectbox("Select Target Variable", available_targets)
    available_features = [col for col in df.columns if col not in ['Entity', 'Year', target]]
    selected_features = st.multiselect("Select Features", available_features, default=available_features[:2])
    model_choice = st.radio("Select Model", ["Random Forest", "XGBoost"], horizontal=True)
    st.markdown("---")
    st.info("Note: All features are standardized before model training.")

# Data Processing
@st.cache_data
def process_data(df, country, features, target):
    filtered_df = df[df['Entity'] == country][['Year'] + features + [target]].dropna()
    return filtered_df, features, target

try:
    filtered_df, selected_features, target = process_data(df, selected_country, selected_features, target)
    if filtered_df.empty:
        st.error("No data available for selected parameters.")
        st.stop()
except Exception as e:
    st.error(f"Data processing error: {str(e)}")
    st.stop()

# Create tabs
tab1, tab2, tab3 = st.tabs(["📊 Data Overview", "🤖 Model Training", "🔮 Predictions"])

with tab1:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Filtered Data Preview")
        st.dataframe(filtered_df.tail(5), use_container_width=True, height=250)
    
    with col2:
        st.subheader("Target Variable Trend")
        fig = px.line(filtered_df, x='Year', y=target, 
                     title=f'{target} Trend in {selected_country}',
                     markers=True, line_shape='spline')
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Feature Correlations")
    corr_matrix = filtered_df[selected_features + [target]].corr()
    fig = px.imshow(corr_matrix, text_auto=".2f", aspect="auto",
                   color_continuous_scale='Blues')
    st.plotly_chart(fig, use_container_width=True)

# Model Training
X = filtered_df[selected_features]
y = filtered_df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def train_model(X_train, y_train, model_type):
    model = None
    if model_type == "Random Forest":
        model = RandomForestRegressor(n_estimators=200, random_state=42)
    elif model_type == "XGBoost":
        model = XGBRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
    
    model.fit(X_train, y_train)
    return model

with tab2:
    try:
        model = train_model(X_train_scaled, y_train, model_choice)
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        # Display metrics in columns
        st.subheader("Model Performance")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="metric-card">📉 **RMSE**<br><h3 style="color:#2e86c1">{:.3f}</h3></div>'.format(rmse), unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card">📈 **R² Score**<br><h3 style="color#2e86c1">{:.3f}</h3></div>'.format(r2), unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-card">📏 **MAE**<br><h3 style="color:#2e86c1">{:.3f}</h3></div>'.format(mae), unsafe_allow_html=True)
        
        # Feature Importance
        if model_choice in ["Random Forest", "XGBoost"]:
            st.subheader("Feature Importance")
            importances = model.feature_importances_
            importance_df = pd.DataFrame({'Feature': selected_features, 'Importance': importances})
            fig = px.bar(importance_df.sort_values('Importance', ascending=True), 
                        x='Importance', y='Feature', orientation='h',
                        color='Importance', color_continuous_scale='Blues')
            st.plotly_chart(fig, use_container_width=True)
            
    except Exception as e:
        st.error(f"Model training failed: {str(e)}")

with tab3:
    try:
        # Create prediction comparison dataframe
        comparison_df = pd.DataFrame({
            'Actual': y_test.values,
            'Predicted': y_pred.flatten(),
            'Year': X_test['Year'] if 'Year' in X_test else X_test.index
        }).sort_index()

        st.subheader("Prediction Analysis")
        
        # Time-based predictions
        if 'Year' in comparison_df:
            fig = px.line(comparison_df, x='Year', y=['Actual', 'Predicted'],
                         title='Actual vs Predicted Values Over Time',
                         markers=True, line_shape='spline')
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = px.scatter(comparison_df, x='Actual', y='Predicted',
                            trendline="ols", title='Actual vs Predicted Values')
            st.plotly_chart(fig, use_container_width=True)
        
        # Residuals plot
        comparison_df['Residuals'] = comparison_df['Actual'] - comparison_df['Predicted']
        fig = px.scatter(comparison_df, x='Predicted', y='Residuals',
                        title='Residual Analysis',
                        trendline="ols", color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Prediction visualization error: {str(e)}")

st.markdown("---")
st.markdown("Climate Change Impact Analyzer v1.0 | Developed by [Calvin]")

