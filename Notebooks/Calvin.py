#!/usr/bin/env python
# coding: utf-8

# ## **Analyzing Global Sea Level Rise: Trends, Clustering, and Impact Assessment**

# ### **Explanation:**
# 
# Sea level rise is one of the most visible and alarming consequences of climate change, largely driven by the increasing concentration of greenhouse gases (GHGs) in the atmosphere. The burning of fossil fuels, deforestation, and industrial activities have led to a significant rise in carbon dioxide (CO₂) emissions, which trap heat in the Earth's atmosphere. This results in the thermal expansion of ocean water and the accelerated melting of glaciers and polar ice caps, contributing to rising sea levels.  
# 
# As sea levels continue to rise, coastal regions worldwide face increased risks of flooding, erosion, and loss of critical infrastructure. Low-lying nations and densely populated coastal cities are particularly vulnerable, with potential economic, social, and environmental consequences.  
# 
# ### **Main Objective:**  
# **To analyze historical and projected sea level rise trends, identify potential anomalies, and assess the impact of CO₂ and greenhouse gas emissions in accelerating these changes.**  
# 
# By understanding these relationships, we can inform policy decisions, advocate for emission reduction strategies, and explore mitigation efforts such as carbon capture technologies and sustainable urban planning to minimize future risks.

# In[45]:


# Required Libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import os

# Modelling
from sklearn.cluster import KMeans
os.environ["OMP_NUM_THREADS"] = "1"
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error




# In[46]:


#Load Datasets

# Define file paths
file3 = r"C:\Users\hp\Desktop\Calvin Desktop\Capstone Project  -  Group 8\Data\Climate Change - datasets\Global_sea_level_rise.csv"

# Load CSV files
Global_sea_level = pd.read_csv(file3)



# In[47]:


Global_sea_level_1 = Global_sea_level.copy()


# In[48]:


Global_sea_level.head()


# In[49]:


# Check the structure of the datasets
print(Global_sea_level.info(), "\n")


# In[50]:


print("Missing values in Global Sea Level Dataset:")
print(Global_sea_level.isnull().sum(), "\n")


# In[51]:


print(Global_sea_level.describe(), "\n")


# In[52]:


print("Duplicates in Global Sea Level Dataset:", Global_sea_level.duplicated().sum())


# 2. Global Sea Level Rise Over Time
# 
# This will visualize how sea levels have changed over time.

# In[53]:


plt.figure(figsize=(12, 6))
sns.lineplot(x=Global_sea_level["year"], y=Global_sea_level["mmfrom1993-2008average"], marker="o", linestyle="-")
plt.axhline(0, color="gray", linestyle="--", alpha=0.7)  # Reference line for 1993-2008 average
plt.xlabel("Year")
plt.ylabel("Sea Level Change (mm)")
plt.title("Global Sea Level Rise Over Time")
plt.grid(True)
plt.show()


# ## **Trend Analysis**
# 
# **Sea Level Rise Overtime: Linear Regression**

# In[54]:


X = Global_sea_level[["year"]]
y = Global_sea_level["mmfrom1993-2008average"]

model = LinearRegression()
model.fit(X, y)

Global_sea_level["predicted"] = model.predict(X)

plt.figure(figsize=(12, 6))
sns.scatterplot(x=Global_sea_level["year"], y=Global_sea_level["mmfrom1993-2008average"], label="Actual Data")
sns.lineplot(x=Global_sea_level["year"], y=Global_sea_level["predicted"], color="red", label="Linear Trend")
plt.xlabel("Year")
plt.ylabel("Sea Level Change (mm)")
plt.title("Sea Level Rise Trend with Linear Regression")
plt.legend()
plt.grid(True)
plt.show()


# ## **Clustering (K-Means)**
# 
# **Identify Patterns in sea level rise across different periods**

# In[55]:


kmeans = KMeans(n_clusters=3, random_state=42)
Global_sea_level["cluster"] = kmeans.fit_predict(Global_sea_level[["mmfrom1993-2008average"]])

plt.figure(figsize=(12, 6))
sns.scatterplot(x=Global_sea_level["year"], y=Global_sea_level["mmfrom1993-2008average"], hue=Global_sea_level["cluster"], palette="viridis")
plt.xlabel("Year")
plt.ylabel("Sea Level Change (mm)")
plt.title("Clustering of Sea Level Rise Data")
plt.legend(title="Cluster")
plt.grid(True)
plt.show()


# The clustering of sea level rise data in the plot above suggests that the sea level change has followed distinct phases over time. Here are some key insights:
# 
# ### **Three Distinct Phases of Sea Level Change**
# The clustering algorithm has grouped the data into three clusters (yellow, purple, and blue), which represent different historical trends in sea level rise.
# 
# - **Cluster 2 (Yellow) - Pre-1940:**  
#   - Represents the earliest period (before 1940), where sea level change was relatively gradual and stable.  
#   - The rate of increase was slow, possibly due to limited human impact on climate during that time.
# 
# - **Cluster 0 (Purple) - 1940 to ~1980:**  
#   - Marks an increase in the rate of sea level rise compared to the previous period.  
#   - This phase could be linked to the start of industrialization’s major impacts on global temperatures, causing glaciers to melt and oceans to warm.
# 
# - **Cluster 1 (Blue) - Post-1980 to Present:**  
#   - Shows a much steeper increase in sea level rise, indicating acceleration.  
#   - This period aligns with increased global warming effects, ice sheet melting, and thermal expansion of seawater.
# 
# ### **Trend Analysis**
# - The clustering suggests that sea level rise has not been uniform but has **accelerated over time**, particularly after 1980.
# - The steep incline in the blue cluster (post-1980) suggests an **exponential increase** in sea level change, possibly due to rising global temperatures and greenhouse gas emissions.
# 
# ### **Implications**
# - The identified clusters reinforce the need for climate action, as recent trends suggest a **rapidly worsening situation**.
# - Policymakers and researchers can use such analysis to **predict future acceleration** and **prepare mitigation strategies**.
# 
# 

# ## **Anomaly Detection**
# 
# **Check for unusual changes in sea levels**

# In[56]:


# Calculate Z-score for anomaly detection
Global_sea_level["z_score"] = (Global_sea_level["mmfrom1993-2008average"] - 
                               Global_sea_level["mmfrom1993-2008average"].mean()) / \
                               Global_sea_level["mmfrom1993-2008average"].std()

# Mark anomalies where the absolute Z-score is greater than 2
Global_sea_level["anomaly"] = Global_sea_level["z_score"].abs() > 2  

# Plot anomalies in sea level rise
plt.figure(figsize=(12, 6))
sns.scatterplot(x=Global_sea_level["year"], 
                y=Global_sea_level["mmfrom1993-2008average"], 
                hue=Global_sea_level["anomaly"], 
                palette={False: "blue", True: "red"})

# Labels and title
plt.xlabel("Year")
plt.ylabel("Sea Level Change (mm)")
plt.title("Anomaly Detection in Sea Level Rise")
plt.grid(True)
plt.show()



# The anomalies detected between 1980 and the present in sea level rise data could be attributed to several key factors:
# 
# ### **Accelerated Global Warming**  
# - Since the 1980s, there has been a significant increase in **global temperatures** due to rising greenhouse gas (GHG) emissions, especially from industrialization, deforestation, and fossil fuel consumption.  
# - Higher temperatures lead to more **thermal expansion** of ocean water, contributing to a faster rise in sea levels.
# 
# ### **Ice Sheet and Glacier Melting**  
# - The **Greenland and Antarctic ice sheets** have been melting at an accelerated rate since the late 20th century.  
# - **Glaciers worldwide** (e.g., in the Himalayas, Alps, and Andes) have also been shrinking, adding to sea level rise.
# 
# ### **Increased Frequency of Extreme Weather Events**  
# - More **frequent and intense hurricanes, storms, and typhoons** have caused storm surges, coastal erosion, and flooding, which could influence sea level data.  
# - **El Niño events** (which cause temporary spikes in sea level) have been more intense in recent decades.
# 
# ### **Anthropogenic Activities Affecting Coastal Regions**  
# - Coastal development and **land subsidence** due to groundwater extraction and urbanization can make sea level rise appear more extreme in certain regions.  
# - **Dams and reservoirs** initially slowed sea level rise, but as reservoirs filled up and land use changed, this effect diminished.
# 
# ### **Changes in Ocean Circulation and Climate Feedback Loops**  
# - Disruptions in **ocean currents** (e.g., weakening of the Atlantic Meridional Overturning Circulation, AMOC) can lead to irregularities in sea level rise.  
# - **Positive feedback loops**, such as the **albedo effect** (less ice means more heat absorption, causing even more melting), have accelerated changes.
# 
# ### **Advancements in Data Collection and Detection Methods**  
# - **Satellite measurements (e.g., TOPEX/Poseidon, Jason-1, Jason-2, Jason-3)** since the 1990s have improved accuracy in sea level rise detection.  
# - Data anomalies might also reflect improved precision rather than sudden shifts.
# 
# ### **Summary**  
# The anomalies between 1980 and the present likely reflect a **combination of human-induced climate change, ice melt acceleration, extreme weather, and improved measurement techniques**. The last few decades have seen a **dramatic increase in sea level rise rates**, and these anomalies may indicate an even steeper upward trend in the coming years.
# 
# 

# ## **The ARIMA-based Forecast of Global Sea Level**

# In[57]:


# Ensure year is the index and convert to datetime format
Global_sea_level_1["year"] = pd.to_datetime(Global_sea_level_1["year"], format="%Y")
Global_sea_level_1.set_index("year", inplace=True)

# Fit an ARIMA model (tune order=(p, d, q) for better accuracy)
model = ARIMA(Global_sea_level_1["mmfrom1993-2008average"], order=(2, 1, 2))  
model_fit = model.fit()

# Forecast the next 30 years
future_steps = 30
future_years = [Global_sea_level_1.index[-1] + pd.DateOffset(years=i) for i in range(1, future_steps + 1)]
forecast = model_fit.forecast(steps=future_steps)

# Convert forecast to a Pandas Series with correct index
forecast_series = pd.Series(forecast, index=pd.to_datetime(future_years))

# Plot historical and predicted values
plt.figure(figsize=(12, 6))
plt.plot(Global_sea_level_1.index, Global_sea_level_1["mmfrom1993-2008average"], label="Actual Sea Level Rise", color="blue")
plt.plot(forecast_series.index, forecast_series, label="Predicted Sea Level Rise", color="red", linestyle="dashed")

# Labels and title
plt.xlabel("Year")
plt.ylabel("Sea Level Change (mm)")
plt.title("Global Sea Level Rise Prediction (ARIMA)")
plt.legend()
plt.grid(True)
plt.show()


# In[67]:


import itertools
import pandas as pd
import statsmodels.api as sm

# Define the range of p, d, q values to test
p = range(0, 4)
d = range(0, 3)
q = range(0, 4)
data = Global_sea_level_1["mmfrom1993-2008average"]

# Generate all possible combinations of p, d, q
pdq_combinations = list(itertools.product(p, d, q))

# Fit ARIMA models and select the best one based on AIC
best_aic = float("inf")
best_order = None
best_model = None

for order in pdq_combinations:
    try:
        model = sm.tsa.ARIMA(data, order=order)
        results = model.fit()
        if results.aic < best_aic:
            best_aic = results.aic
            best_order = order
            best_model = results
    except:
        continue  # Skip combinations that fail

print(f"Best ARIMA Order: {best_order}")


# Test CODE for tuning

# In[58]:


import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm

# Define the range of p, d, q values to test
p = range(0, 4)  # Typically, 0 to 3
d = range(0, 3)  # Differencing: 0, 1, or 2
q = range(0, 4)  # Typically, 0 to 3

# Generate all possible combinations
pdq_combinations = list(itertools.product(p, d, q))

# Load your dataset (assuming 'year' is the index and data column is 'mmfrom1993-2008average')
data = Global_sea_level_1["mmfrom1993-2008average"]

# Grid search to find the best ARIMA order
best_aic = np.inf  # Set initial AIC to infinity
best_order = None
results = []

for order in pdq_combinations:
    try:
        model = sm.tsa.ARIMA(data, order=order)  # Fit model
        model_fit = model.fit()
        aic = model_fit.aic  # Get AIC score
        
        results.append((order, aic))

        # Update best order if a lower AIC is found
        if aic < best_aic:
            best_aic = aic
            best_order = order
            
    except:
        continue  # Skip models that fail

# Display best order
print(f"Best ARIMA Order: {best_order} with AIC: {best_aic}")

# Show top 5 best models
results.sort(key=lambda x: x[1])
for order, aic in results[:5]:
    print(f"ARIMA{order} - AIC: {aic}")


# ## **Tuning Parameters**

# **The Best Model: Based on the above AIC(Akaic Information Criterion) results is ARIMA(0,2,2)**
# 
# A lowe AIC means a good model 
# 
# p=0 → No autoregressive terms → Past values do not directly impact the future.
# 
# d=2 → The data was differenced twice to remove trends and make it stationary.
# 
# q=2 → The model accounts for the last two forecast errors to improve predictions.
# 
# The best forecast strategy in this case is to look at the trend and use past forecast errors to adjust future predictions.

# In[59]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

def forecast_sea_level(data, date_col, value_col, order=(0, 2, 2), future_years=30):
    """
    Fits an ARIMA model to forecast future sea level changes.

    Parameters:
    - data (DataFrame): The dataset containing historical sea level data.
    - date_col (str): Column name for the date (must be convertible to datetime).
    - value_col (str): Column name for the sea level measurement.
    - order (tuple): ARIMA (p, d, q) parameters. The model with the lowest AIC is (0,2,2).
    - future_years (int): Number of future years to forecast. Default is 30.

    Returns:
    - forecast_df (DataFrame): DataFrame with future dates and predicted values.
    """

    # Ensure the date column is in datetime format
    data[date_col] = pd.to_datetime(data[date_col])
    data.set_index(date_col, inplace=True)

    # Ensure no missing values
    data = data[[value_col]].dropna()

    # Fit ARIMA model
    try:
        model = ARIMA(data[value_col], order=order)
        model_fit = model.fit()

        # Generate future dates
        last_date = data.index[-1]
        future_dates = [last_date + pd.DateOffset(years=i) for i in range(1, future_years + 1)]
        forecast = model_fit.forecast(steps=future_years)

        # Convert forecast into DataFrame
        forecast_df = pd.DataFrame({date_col: future_dates, value_col: forecast.values})
        forecast_df[date_col] = pd.to_datetime(forecast_df[date_col])

        # Plot actual vs predicted values
        plt.figure(figsize=(12, 6))
        plt.plot(data.index, data[value_col], label="Actual Sea Level Rise", color="blue")
        plt.plot(forecast_df[date_col], forecast_df[value_col], label="Predicted Sea Level Rise", color="red", linestyle="dashed")
        plt.xlabel("Year")
        plt.ylabel("Sea Level Change (mm)")
        plt.title("Global Sea Level Rise Prediction (ARIMA)")
        plt.legend()
        plt.grid(True)
        plt.show()

        return forecast_df

    except Exception as e:
        print(f"Error: {e}")
        return None

# Example usage
forecast_results = forecast_sea_level(Global_sea_level, "date", "mmfrom1993-2008average")



# ## **Model Validation**

# In[60]:


#Residuals (Errors) Should Look Like White Noise

#If the model fits well, the errors should not show any patterns.

residuals = model_fit.resid
plt.figure(figsize=(10,4))
plt.plot(residuals)
plt.title("Residuals of ARIMA(1,2,2)")
plt.show()


# **Interpretatiion of Residuals**
# 
# Residuals = (Actual Values - Predicted Values).
# 
# The above plot shows that residuals are randomly scattered around zero, indicating that the model has captured the data trend well.

# In[62]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

# Load and preprocess the data
data = Global_sea_level_1.copy()

# Ensure the 'date' column is in datetime format
data["date"] = pd.to_datetime(data["date"])

# Extract the year from the 'date' column
data["year"] = data["date"].dt.year

# Set the year as the index (keeping it as a time-based index)
data.set_index("date", inplace=True)

# Fit ARIMA(0,2,2) model
model = ARIMA(data["mmfrom1993-2008average"], order=(0, 2, 2))
model_fit = model.fit()

# Extract residuals
residuals = model_fit.resid

# Plot ACF and PACF
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

plot_acf(residuals, ax=axes[0], title="ACF of Residuals")
plot_pacf(residuals, ax=axes[1], title="PACF of Residuals")

plt.show()

# Check residual normality with histogram and QQ plot
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].hist(residuals, bins=20, edgecolor="black", alpha=0.7)
ax[0].set_title("Histogram of Residuals")

sm.qqplot(residuals, line="s", ax=ax[1])
ax[1].set_title("QQ Plot of Residuals")

plt.show()


# **Interpretation**
# 
# ✅ The model predicts a significant rise in sea levels over the next 30 years.
# 
# ✅ The rate of increase is consistent with past observations, meaning sea levels are expected to continue rising at a similar pace.
# 
# ✅ The accuracy of the forecast depends on whether past trends continue—external factors like climate interventions, ice melt acceleration, or changing ocean currents could alter this trend.

# The ARIMA-based forecast of global sea level rise suggests a continued and accelerating increase beyond 2020. The trend shown in the historical data aligns with real-world observations of rising sea levels due to climate change.
# 
# ### **Key Insights from the Prediction**:
# 1. **Consistent Upward Trend** – The data shows that sea levels have been rising steadily since the late 19th century, with an increasing rate in recent decades.
# 2. **Projected Acceleration** – The red dashed line indicates that if current trends continue, sea levels will continue to rise significantly, potentially reaching 150mm above the baseline by 2050.
# 3. **Climate Change Impact** – This rise is largely attributed to global warming, causing:
#    - **Glacial and Ice Sheet Melting** – Particularly in Greenland and Antarctica.
#    - **Thermal Expansion** – As ocean water warms, it expands.
#    - **Increased Coastal Flooding & Erosion** – More frequent and severe flooding events, leading to displacement and habitat loss.
# 

# If sea levels continue to rise significantly above the baseline, the world will undergo drastic changes, affecting multiple aspects of life. Below is an overview of how different sectors will be impacted:
# 
# ---
# 
# ## **How the World Will Look with Rising Sea Levels**
# 
# ---
# 
# ### **Economic Impact**
# 🔹 **Loss of Infrastructure & Property Damage**  
#    - Coastal cities like **New York, Mumbai, Jakarta, and London** will face severe flooding, leading to billions in damages.
#    - Ports, airports, and industrial zones will be at risk, disrupting trade.
#    - Insurance costs will **skyrocket**, with rising premiums for flood-prone areas.
#   
# 🔹 **Displacement & Job Loss**  
#    - Fishing, tourism, and real estate industries will suffer.
#    - Coastal farms and industries will relocate, causing **mass unemployment** in affected areas.
#    - **Rural-to-urban migration** will strain inland cities.
# 
# 🔹 **Agriculture & Food Security**  
#    - **Saltwater intrusion** will destroy fertile land, reducing global crop yields.
#    - Fisheries will collapse as **ocean acidification** and changing currents disrupt marine ecosystems.
# 
# ---
# 
# ### **Health Impacts**
# 🔹 **Waterborne Diseases**  
#    - Flooding increases **cholera, malaria, dengue, and typhoid** outbreaks.
#    - Standing water leads to mosquito breeding, spreading **Zika and malaria**.
# 
# 🔹 **Heat Stress & Respiratory Issues**  
#    - Rising temperatures worsen **heat strokes, respiratory diseases**, and cardiovascular problems.
#    - **Air pollution from wildfires** will intensify, affecting millions.
# 
# 🔹 **Mental Health Crisis**  
#    - Climate refugees will experience **anxiety, PTSD, and depression** due to displacement and loss of homes.
# 
# ---
# 
# ### **Demographic & Social Life Changes**
# 🔹 **Mass Migration (Climate Refugees)**  
#    - **Over 1 billion people** could be displaced by 2050, leading to resource conflicts.
#    - Small island nations (Maldives, Tuvalu, Kiribati) may become **uninhabitable**.
#    - Inland areas will face **overpopulation, housing crises, and social unrest**.
# 
# 🔹 **Cultural Heritage Loss**  
#    - Historical sites like Venice, Machu Picchu, and the Great Pyramids could be endangered.
#    - Indigenous communities relying on coastal ecosystems will **lose their way of life**.
# 
# ---
# 
# ### **Urban Planning Challenges**
# 🔹 **Redesigning Cities for Rising Waters**  
#    - Countries will need **floating cities, seawalls, and elevated infrastructure** (e.g., Netherlands’ flood-resistant designs).
#    - Underground transportation systems like **subways and tunnels will become obsolete** in some cities.
#    - Governments will need to **relocate entire populations**, requiring massive funding.
# 
# 🔹 **Costly Adaptation Efforts**  
#    - Billions of dollars will be spent on **drainage systems, water barriers, and artificial islands**.
#    - Some cities will need to **abandon low-lying areas**, leading to financial losses.
# 
# ---
# 
# ### **Weather & Climate Impact**
# 🔹 **More Extreme Storms & Hurricanes**  
#    - Warmer oceans fuel **super typhoons and hurricanes**, leading to catastrophic flooding.
#    - Wind patterns will shift, increasing droughts in some areas and heavy rains in others.
# 
# 🔹 **Longer & More Intense Heatwaves**  
#    - Higher temperatures will disrupt **agriculture, power grids, and water supplies**.
#    - Wildfires will worsen, destroying millions of acres annually.
# 
# 🔹 **Disrupted Ocean Currents**  
#    - The melting Arctic could weaken the **Atlantic Meridional Overturning Circulation (AMOC)**, disrupting global weather.
#    - Areas reliant on monsoons (India, Southeast Asia) may face **unpredictable rainfall**.
# 
# ---
# 
# ### **The Urgent Need for Action**
# Sea level rise is not a distant threat—it is happening **now**. Governments, businesses, and individuals must **invest in climate resilience**, reduce carbon emissions, and prepare for an era of change. 
# 
# 
