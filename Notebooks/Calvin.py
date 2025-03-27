#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import explained_variance_score, max_error

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# Configure page
st.set_page_config(page_title="Climate Analyzer", layout="wide", page_icon="🌍")

# Custom CSS for professional look
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    .metric-card {
        padding: 20px;
        background: white;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
        margin: 10px 0;
    }
    .metric-header {
        color: #2c3e50 !important;
        font-size: 1.1rem !important;
        margin-bottom: 8px !important;
    }
    .metric-value {
        color: #2e86c1 !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }
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
    if model_type == "Random Forest":
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5]
        }
        model = GridSearchCV(RandomForestRegressor(random_state=42),
                           param_grid, cv=TimeSeriesSplit(3), scoring='neg_root_mean_squared_error')
        
    elif model_type == "XGBoost":
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5],
            'subsample': [0.8, 1.0]
        }
        model = GridSearchCV(XGBRegressor(random_state=42),
                           param_grid, cv=TimeSeriesSplit(3), scoring='neg_root_mean_squared_error')
    
    model.fit(X_train, y_train)
    st.success(f"Best params: {model.best_params_}")
    return model.best_estimator_

# Add forecast controls in sidebar
with st.sidebar:
    st.markdown("---")
    forecast_years = st.number_input("Forecast Horizon (years)", 5, 50, 30)
    confidence_level = st.slider("Confidence Interval", 0.7, 0.99, 0.9)

with tab2:
    try:
        model = train_model(X_train_scaled, y_train, model_choice)
        y_pred = model.predict(X_test_scaled)
        
        # Enhanced Metrics
        metrics = {
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'MAE': mean_absolute_error(y_test, y_pred),
            'R²': r2_score(y_test, y_pred),
            'Explained Variance': explained_variance_score(y_test, y_pred),
            'Max Error': max_error(y_test, y_pred),
            'MAPE': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        }
        
        # Display metrics in columns
        st.subheader("Model Diagnostics")
        cols = st.columns(3)
        
        # Create metric cards in grid
        for i, (name, value) in enumerate(metrics.items()):
            with cols[i % 3]:  # Cycle through 3 columns
                st.markdown(f'''
                    <div class="metric-card">
                        <div class="metric-header">📊 {name}</div>
                        <div class="metric-value">{
                            f'{value:.3f}%' if name == 'MAPE' 
                            else f'{value:.3f}'
                        }</div>
                    </div>
                ''', unsafe_allow_html=True)
        
        # Feature Importance with permutation importance
        st.subheader("Feature Analysis")
        with st.spinner("Calculating feature importance..."):
            result = permutation_importance(
                model, 
                X_test_scaled, 
                y_test, 
                n_repeats=10, 
                random_state=42
            )
            
            importance_df = pd.DataFrame({
                'Feature': selected_features,
                'Importance': result.importances_mean,
                'Std Dev': result.importances_std
            }).sort_values('Importance', ascending=True)

            fig = px.bar(
                importance_df, 
                x='Importance', 
                y='Feature', 
                orientation='h',
                error_x='Std Dev', 
                color='Importance', 
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Model training failed: {str(e)}")
        st.error("Please check your feature/target selection and data quality")

with tab3:
    try:
        # Generate future timeline
        last_year = filtered_df['Year'].max()
        future_years = pd.DataFrame({'Year': range(last_year + 1, last_year + forecast_years + 1)})
        
        # Create future features (assuming simple trend extension)
        future_X = pd.DataFrame({
            feature: np.linspace(filtered_df[feature].iloc[-1], 
                               filtered_df[feature].iloc[-1] * 1.5, 
                               num=forecast_years)
            for feature in selected_features
        })
        
        future_X_scaled = scaler.transform(future_X)
        future_pred = model.predict(future_X_scaled)
        
        # Create combined timeline
        full_timeline = pd.concat([
            filtered_df[['Year', target]].rename(columns={target: 'Actual'}),
            pd.DataFrame({'Year': future_years['Year'], 'Predicted': future_pred})
        ])
        
        # Calculate growth rate
        current_value = filtered_df[target].iloc[-1]
        future_value = future_pred[-1]
        growth_pct = ((future_value - current_value) / current_value) * 100
        
        # Plot extended predictions
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=full_timeline['Year'], y=full_timeline['Actual'],
            name='Historical Actual', mode='lines+markers'
        ))
        fig.add_trace(go.Scatter(
            x=full_timeline['Year'], y=full_timeline['Predicted'], 
            name='Model Predictions', mode='lines+markers',
            line=dict(dash='dot')
        ))
        fig.add_vline(x=last_year, line_dash="dash", line_color="red",
                    annotation_text="Current Year", annotation_position="top left")
        
        fig.update_layout(
            title=f"{target} Projections through {last_year + forecast_years}",
            xaxis_title='Year',
            yaxis_title=target,
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Display growth metrics
        st.subheader("Climate Change Impact Summary")
        growth_cols = st.columns(2)
        with growth_cols[0]:
            st.metric("Current Value (Last Observed)", f"{current_value:.2f}")
            st.metric("Projected Value", f"{future_value:.2f} (±{future_pred.std():.2f})")
        with growth_cols[1]:
            st.metric("Absolute Change", f"{future_value - current_value:.2f}")
            st.metric("Percentage Growth", f"{growth_pct:.1f}%")
            
    except Exception as e:
        st.error(f"Projection error: {str(e)}")

st.markdown("---")
st.markdown("Climate Change Impact Analyzer v1.0 | Developed by Calvin")

