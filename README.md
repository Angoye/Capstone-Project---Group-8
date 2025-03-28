# Climate Foresight: Predicting CO₂ Emissions, Temperature Rise, and Sea Level Trends Using Machine Learning

# Elevator Pitch
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




 
