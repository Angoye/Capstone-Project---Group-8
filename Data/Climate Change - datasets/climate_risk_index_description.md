# 🌍 Climate Risk Index Dataset

This dataset captures country-level exposure and vulnerability to extreme weather events. It combines key climate-related impact metrics like fatalities, losses, and rankings based on the Climate Risk Index (CRI).

## 📊 Dataset Overview

- **Rows**: 182
- **Columns**: 17
- **Scope**: Global — each row represents a country

---

## 🧾 Column Descriptions

### 🌐 Geographic Metadata
- `index`: Row index
- `cartodb_id`: Internal identifier (possibly from Carto platform)
- `the_geom`: Geospatial geometry (WKT or GeoJSON)
- `the_geom_webmercator`: Web Mercator projection of `the_geom`

### 🏳️ Country Info
- `country`: Name of the country
- `rw_country_code`: ReliefWeb country code
- `rw_country_name`: ReliefWeb country name

### 🔢 Climate Risk Metrics
- `cri_rank`: Country's overall Climate Risk Index rank (lower is higher risk)
- `cri_score`: CRI score — a composite indicator of climate-related impacts

### ⚰️ Human Impact
- `fatalities_total`: Total number of deaths due to extreme climate events
- `fatalities_rank`: Rank based on number of fatalities
- `fatalities_per_100k_total`: Deaths per 100,000 people
- `fatalities_per_100k_rank`: Rank based on per capita fatalities

### 💸 Economic Losses
- `losses_usdm_ppp_total`: Estimated total losses (in million USD, adjusted for PPP)
- `losses_usdm_ppp_rank`: Rank based on absolute losses
- `losses_per_gdp__total`: Losses as a percentage of GDP
- `losses_per_gdp__rank`: Rank based on relative economic impact

---

## 🔍 Use Cases

- Identify high-risk countries based on climate vulnerability
- Correlate fatalities, economic losses, and country profiles
- Feed into risk models or climate impact assessments
- Support for funding and aid decision-making

---

> ⚠️ Note: Geometric columns (`the_geom`, `the_geom_webmercator`) may require geospatial processing tools like GeoPandas or GIS platforms.

## 📜 License
Specify the data source and license (e.g., [Germanwatch](https://www.germanwatch.org/en/cri), Creative Commons, etc.)
