# Climate Foresight: Predicting CO₂ Emissions, Temperature Rise, and Sea Level Trends Using Machine Learning

# Overview
> Climate change is a global crisis yet many regions lack reliable forecasts to guide policy and investments.By leveraging historical climate and socioeconomic data,the project uses Machine Learning and Time Series forecasting to predict critical climate indicators:CO₂ emissions per capita, average annual temperature, and global sea level rise.
> By transforming Raw data into actionable climate intelligence,our model empowers Policy makers design effective policies, enables investors to anticipate risks and facilitates resource allocation turning data into proactive climate solution.

 ![Uploading image.png…]()
  
# Business Understanding and Data Understanding
> Climate change poses significant risk to economies,ecosystem and Health.Government and Organisations rely on accurate climate projections to implement mitigation strategies.

> The Climate Risk Index evaluates vulnerability to extreme weather events and climate-related risks. It ranks countries based on climate-related fatalities, economic losses, and exposure to hazards. According to Our World in Data (Ritchie et al., 2023), nations with lower Climate Risk Index (CRI) ranks face higher risks, particularly in regions with weak adaptive infrastructure

> According to data from Our World in Data (Ritchie et al., 2023) highlights the accelerating trends in CO₂ emissions and their direct correlation with rising global temperatures. Additionally,nations with lower Climate Risk Index (CRI) ranks face higher risks, particularly in regions with weak adaptive infrastructure.

>  These findings reinforce the urgency of developing predictive models to aid policymakers, businesses, and environmental agencies in designing more effective interventions for emissions reduction, climate resilience, and disaster preparedness.

> This project aims to build a predictive model that forecasts:

     - Annual CO₂ emissions per capita  
     
     - Average annual temperature
     
     - Global sea level rise
     
> By providing both short-term predictions and long-term forecasts, this project bridges the gap between raw data and actionable climate intelligence.

> # Data understanding

> Climate data final dataset:

> Source:  [Our World in Data](https://ourworldindata.org/), [World Bank](https://data.worldbank.org/).

> Size: 6,323 records with 28 features.

> Purpose: Tracks CO₂ and GHG emissions by country and sector over time.

> Key Features:

 - Sector-wise emissions: Tracks emissions from transport, industry, electricity, and more.

 - Annual CO₂ emissions per capita: Reflects individual contributions to emissions.
  
 - Average temperature: Yearly average regional/country temperatures.
  
 - Sea level rise: Measured relative to the 1993–2008 baseline.
  
 - Forest area (% of land): Indicates national forest coverage.
  
- Renewable energy usage: Share of energy derived from renewable sources.
  
Climate Risk Index:

> Source: [Our World in Data](https://ourworldindata.org/).

> Size: 182 records with 17 features

> Purpose: Evaluates vulnerability to extreme weather events and climate-related risks.

> Key Features:

 - Climate Risk Index Rank (cri_rank): Indicates relative climate risk (lower = higher risk).

 - Fatalities (fatalities_total): Deaths caused by extreme weather.

 - Economic Losses (losses_usdm_ppp_total): Monetary losses adjusted for PPP (in USD millions).

 - Losses per GDP (losses_per_gdp_total): Reflects economic vulnerability to climate shocks.

 
# Modeling and Evaluation

> Models used:

 - Supervised Learning: Random Forest, XGBoost, Linear Regression (for CO₂ emissions and temperature prediction)

 - Time Series Models: 

   
 > Model Performance:

 - XGBoost (non-tuned) performed best, explaining ~97.7% of the variance in CO₂ emissions.

 - Random Forest performed similarly well but slightly below XGBoost.

 - Linear Regression significantly underperformed due to the non-linear nature of the relationships.

 > Surprisingly, tuning slightly reduced performance for both models.

     -Possible reasons: overfitting on training data.

     - Tuned Random Forest outperformed tuned XGBoost in both R² and RMSE.
     
  > In Temperature predictions,Tuned XGBoost was the best model, achieving the lowest RMSE (0.983) and highest R² (98.34%).
 
> Model Performance vs. Baseline:

 - Baseline Model: A simple historical average or linear regression was used as the baseline.

 - Final Model: XGBoost (for emissions and temperature).XGBoost significantly outperformed Linear Regression, capturing non-linear relationships in emissions and 
    temperature trends.It had R² score (0.977) with Lowest RMSE (1.06).

# Conclusion

> Recommended Use of the Model:

 - Policy Planning: Governments can use forecasts to design carbon reduction policies.

 - Disaster Preparedness: Coastal cities and agencies can anticipate flooding risks.

 - Financial Decision-Making: Investors and insurers can assess economic vulnerabilities.

# Recommendations

- Governments should use predictive models to proactively implement carbon tax policies and incentives for green energy adoption.

- Businesses can utilize emissions forecasts to implement sustainable practices and align their operations with global carbon neutrality objectives.

- Further research should be conducted to enhance model accuracy using additional climate variables and ensemble learning techniques.

# Challenges and Limitations

 - Inconsistent data reporting across different sources

 - Uncertainty in long-term forecasts due to external factors (policy changes, natural disasters)




 
