#!/usr/bin/env python
# coding: utf-8

# In[26]:


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
from typing import Dict

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# Configure page
st.set_page_config(page_title="Climate Analyzer", layout="wide", page_icon="🌍")

# Custom CSS for professional look
st.markdown("""
    <style>
    /* Base background */
    .main {background-color: #f8f9fa;}
    
    /* Metric cards */
    .metric-card {
        padding: 20px;
        background: white;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
        margin: 10px 0;
    }
    
    /* Card headers */
    .metric-header {
        color: #2c3e50;
        font-size: 1.1rem;
        margin-bottom: 8px;
        font-family: 'Arial', sans-serif;
    }
    
    /* Values */
    .metric-value {
        color: #2e86c1;
        font-size: 1.8rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    /* Expander headers */
    div[data-testid="stExpander"] details summary p {
        font-size: 1.2rem !important;
        font-weight: 600 !important;
        color: #1a5276 !important;
    }
    
    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .metric-value {
            font-size: 1.4rem;
        }
        div[data-testid="stExpander"] details summary p {
            font-size: 1rem !important;
        }
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
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Data Overview", "🤖 Model Training", "🔮 Predictions","📝 Interpretation","📑 Policy Notes"])

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

# Interpretation generation function
def generate_interpretation(metrics: Dict, model_type: str, target: str) -> str:
    interpretation = f"""
    ##  {target.replace('_', ' ').title()} Model Interpretation
    
    ### Core Insights
    **Model Type**: {model_type}
    
    - **Accuracy Profile**: """
    
    # R² Analysis
    if metrics['R²'] < 0.3:
        interpretation += f"Low explanatory power (R²={metrics['R²']:.3f}) but "
    else:
        interpretation += f"Reasonable explanatory power (R²={metrics['R²']:.3f}) with "
    
    # MAPE Analysis
    if metrics['MAPE'] < 5:
        interpretation += f"excellent relative precision (±{metrics['MAPE']:.1f}% error). "
    else:
        interpretation += f"moderate relative precision (±{metrics['MAPE']:.1f}% error). "
    
    # Error Analysis
    interpretation += f"""
    - **Error Profile**: Typical error of {metrics['RMSE']:.2f} units (MAE={metrics['MAE']:.2f}),
      with worst-case error of {metrics['Max Error']:.2f} units.
    
    ### Climate Context
    For {target.replace('_', ' ')}:
    - ±{metrics['MAPE']:.1f}% error represents """
    
    # Domain-specific examples
    if "temp" in target.lower():
        interpretation += f"approximately ±{0.3*metrics['MAPE']:.1f}°C variance"
    elif "precip" in target.lower():
        interpretation += f"about ±{metrics['MAPE']:.1f}mm rainfall variance"
    else:
        interpretation += "significant variance in measured values"
    
    interpretation += """
    \n- Best used for identifying multi-year trends rather than annual variations
    - Consider combining with domain expertise for policy decisions"""
    
    return interpretation

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

        if 'Predicted_Growth' not in st.session_state:
        # Calculate emission growth rate from predictions
            growth_rate = (y_pred[-1] - y_pred[0]) / (X_test['Year'].max() - X_test['Year'].min())
        st.session_state.Predicted_Growth = growth_rate

        # Save metirics for Interpretation
        st.session_state.metrics = metrics
        st.session_state.model_choice = model_choice  # Store model type
        st.session_state.target = target  # Store selected variable
        
        # Verify storage (temporary debug)
        #st.write("Debug - Stored Metrics:", st.session_state.metrics)


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

        if not X_test.empty and 'Year' in X_test.columns:
            time_span = X_test['Year'].max() - X_test['Year'].min()
            growth_rate = (y_pred[-1] - y_pred[0]) / time_span
            st.session_state.Predicted_Growth = growth_rate
        else:
            # Fallback using filtered_df
            test_years = filtered_df.loc[X_test.index, 'Year']
            time_span = test_years.max() - test_years.min()
            growth_rate = (y_pred[-1] - y_pred[0]) / time_span
            st.session_state.Predicted_Growth = growth_rate       
        
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
with tab4:
    if 'metrics' in st.session_state:
        st.markdown(generate_interpretation(
            st.session_state.metrics,
            st.session_state.model_choice,  # Changed from model_choice
            st.session_state.target         # Changed from target
        ), unsafe_allow_html=True)
        
        # Visual explanation
        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(values=[st.session_state.metrics['R²'], 1 - st.session_state.metrics['R²']],
                        names=['Explained Variance', 'Unexplained'],
                        title=f"R² Breakdown ({st.session_state.metrics['R²']:.1%})",
                        color_discrete_sequence=['#2e86c1', '#e0e0e0'])
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            error_df = pd.DataFrame({
                'Error Type': ['Typical (MAE)', 'Worst Case'],
                'Value': [st.session_state.metrics['MAE'], 
                         st.session_state.metrics['Max Error']]
            })
            fig = px.bar(error_df, x='Error Type', y='Value',
                        title='Error Magnitude Comparison',
                        color='Error Type', 
                        color_discrete_sequence=['#2e86c1', '#28a745'])
            st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
        st.markdown("""
            ### Recommendations
            <div class="metric-card">
                <div class="metric-header">✅ Do</div>
                <ul>
                    <li>Use for multi-year trend analysis</li>
                    <li>Combine with other climate indicators</li>
                    <li>Monitor error distribution quarterly</li>
                </ul>
            </div>
            
            <div class="metric-card">
                <div class="metric-header">⚠️ Don't</div>
                <ul>
                    <li>Rely solely for annual predictions</li>
                    <li>Use for extreme event forecasting</li>
                    <li>Compare directly with raw sensor data</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("Train a model first to see interpretation")

with tab5:
    st.header("Climate Modeling & Policy Guidance")
    
    # Domain Knowledge Section
    with st.expander("Why Climate Models Are Different", expanded=True):
        st.markdown("""
        **Key Climate Modeling Nuances:**
        - 🎯 *Low R² Significance*: A 0.2 R² in climate models can represent meaningful trends due to:
          - Long-term cumulative effects (small annual changes → big decadal impacts)
          - High system complexity (many interacting variables)
          - Measurement uncertainties in historical data
        
        - 📉 *Error Interpretation*:
          ```python
          # Climate impact multiplier
          def climate_impact(error, years=10):
              return error * years * 1.5  # Non-linear amplification
          ```
          - Example: 0.3°C annual error → 4.5°C decade error using 1.5x amplification factor
        
        - 🌡️ *Threshold Effects*: Small errors matter at critical points:
          """)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Ice Melt Threshold", "0°C", "±0.5°C Error Margin")
        with col2:
            st.metric("Crop Failure", "2°C Change", "±1°C Model Uncertainty")
    
    # Carbon Tax Calculator
    with st.expander("Carbon Footprint Tax Projections", expanded=True):
        st.subheader("Emission-Based Tax Estimator")
        
        # User Inputs
        col1, col2, col3 = st.columns(3)
        with col1:
            base_emissions = st.number_input("Current Emissions (MtCO2/yr)", min_value=1.0, value=100.0)
        with col2:
            tax_rate = st.slider("Tax Rate ($/tCO2)", 10, 100, 50)
        with col3:
            projection_years = st.slider("Projection Years", 5, 50, 30)
        
        # Get model predictions if available
        if 'metrics' in st.session_state:
            pred_growth = st.session_state.metrics.get('Predicted_Growth', 0.02)  # Assume 2% annual growth
        else:
            pred_growth = st.slider("Annual Emission Growth Rate", 0.0, 0.1, 0.02)
        
        # Tax Calculation Formula
        years = np.arange(projection_years)
        emissions = base_emissions * (1 + pred_growth) ** years
        tax_liability = emissions * tax_rate
        
        # Create projections dataframe
        tax_df = pd.DataFrame({
            'Year': pd.date_range(start=pd.Timestamp.today(), periods=projection_years, freq='Y').year,
            'Emissions': emissions,
            'Tax': tax_liability
        })
        
        # Display results
        fig = px.area(tax_df, x='Year', y='Tax', 
                     title=f"Projected Tax Liability @ ${tax_rate}/tCO2",
                     labels={'Tax': 'Annual Tax ($B)'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Policy Recommendations
        st.markdown("""
        **Tax Policy Guidance:**
        - Base Rate Formula:
          ```math
          Tax_{t} = Emissions_{t} × (Base Rate + (Predicted ΔT × $100/°C))
          ```
        - Recommended Components:
          1. **Prediction-Based Surcharge**: 
             - $100/°C of projected warming above 1.5°C target
          2. **Error Margin Buffer**:
             - Reserve 20% of tax revenue for prediction uncertainty
          3. **Threshold Penalties**:
             - 2× tax rate for crossing climate thresholds
        """)
    
    # Model Limitations Disclaimer
    st.markdown("""
    ---
    **Critical Assumptions:**
    - Linear emission growth projections (real-world may vary)
    - Constant tax rate policy (actual rates may escalate)
    - Does not account for carbon sequestration efforts
    - Based on {} model accuracy (±{}%)
    """.format(st.session_state.get('model_choice', 'current'), 
    st.session_state.metrics.get('MAPE', 1.1) if 'metrics' in st.session_state else 1.1))    
    st.markdown("---")
    st.markdown("Climate Change Impact Analyzer v1.0 | Developed by Calvin")

