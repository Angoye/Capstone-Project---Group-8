#!/usr/bin/env python
# coding: utf-8

# In[92]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

st.write(f"Seaborn version: {sns.__version__}")

# Load Data
file_path = r"C:\Users\hp\Desktop\Calvin Desktop\Capstone Project  -  Group 8\Data\Climate Change - datasets\climate_data_final_df.csv"
df = pd.read_csv(file_path)

# Drop Unnecessary Column
df.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')

# Streamlit UI
st.title("Climate Change & Energy Impact Analysis")
st.subheader("Exploratory Data Analysis")

# Country and Feature Selection
selected_countries = st.multiselect("Select Countries", df['Entity'].unique(), default=df['Entity'].unique())
selected_features = st.multiselect("Select Features", df.columns[2:], default=df.columns[2:])

df = df[df['Entity'].isin(selected_countries)]

st.write("### Dataset Overview")
st.write(df[selected_features].head())

st.write("### Summary Statistics")
st.write(df[selected_features].describe())

st.write("### Missing Values")
st.write(df[selected_features].isnull().sum())

# Handle Missing Values
df.dropna(inplace=True)  # Can be replaced with df.fillna(df.mean()) if needed

# Feature and Target Selection
targets = ['Average Temperature', 'mmfrom1993-2008average', 'Renewable energy consumption (% of total final energy consumption)']
features = [col for col in selected_features if col not in targets]  # Prevent feature leakage

# Train Model Function
def train_model(X, y, model_type='random_forest'):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    results = {}
    
    for target in y.columns:
        y_train_target = y_train[target]
        y_test_target = y_test[target]
        
        if model_type == 'random_forest':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train_target)
            y_pred_target = model.predict(X_test)
        elif model_type == 'lstm':
            X_train_exp = np.expand_dims(X_train, axis=-1)
            X_test_exp = np.expand_dims(X_test, axis=-1)
            
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(1)
            ])
            
            model.compile(optimizer='adam', loss='mse')
            model.fit(X_train_exp, y_train_target, epochs=50, batch_size=16, verbose=0)
            y_pred_target = model.predict(X_test_exp).flatten()
        
        mae = mean_absolute_error(y_test_target, y_pred_target)
        rmse = np.sqrt(mean_squared_error(y_test_target, y_pred_target))
        r2 = r2_score(y_test_target, y_pred_target)
        
        results[target] = {
            'model': model, 'y_test': y_test_target, 'y_pred': y_pred_target,
            'mae': mae, 'rmse': rmse, 'r2': r2
        }
    
    return results, scaler, X_train, X_test

# Model Selection
model_choice = st.selectbox("Select Model", ["Random Forest", "LSTM"])
X = df[features]
y = df[targets]

results, scaler, X_train, X_test = train_model(X, y, model_type='random_forest' if model_choice == "Random Forest" else 'lstm')

# Display Metrics
st.write("### Model Performance")
for target, res in results.items():
    st.write(f"**{target}** - MAE: {res['mae']:.2f}, RMSE: {res['rmse']:.2f}, R2: {res['r2']:.2f}")

# Feature Importance (Only for Random Forest)
if model_choice == "Random Forest":
    st.write("### Feature Importance")
    importance_df = pd.DataFrame({'Feature': features, 'Importance': results[targets[0]]['model'].feature_importances_})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    fig, ax = plt.subplots()
    sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
    st.pyplot(fig)

# Residual Plot
st.write("### Residual Analysis")
fig, ax = plt.subplots()
for target, res in results.items():
    sns.scatterplot(x=res['y_test'].values, y=(res['y_test'].values - res['y_pred']), ax=ax, label=target)
ax.axhline(y=0, color='r', linestyle='--')
ax.set_xlabel("Actual Values")
ax.set_ylabel("Residuals")
ax.legend()
st.pyplot(fig)


