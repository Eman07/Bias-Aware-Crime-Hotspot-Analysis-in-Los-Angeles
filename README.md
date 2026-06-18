# Bias-Aware Crime Hotspot Analysis in Los Angeles

## Overview
An unsupervised machine learning pipeline that identifies crime hotspots across Los Angeles using K-means clustering with fairness weighting. Integrates crime reports, LAPD station geolocation data, and U.S. Census income data to surface geographic disparities in crime concentration and policing accessibility.

## Key Features
- Multi-source data pipeline integrating LAPD crime reports, police station locations, and ACS Census income data
- K-means clustering with fairness weighting to reduce socioeconomic bias in hotspot detection
- Haversine formula for computing proximity-based spatial features between incidents and police stations
- Visual analytics highlighting geographic crime clusters and arrest rate disparities by neighborhood

## Tech Stack
- **Languages:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- **Data Sources:** LAPD Open Data, U.S. Census Bureau (ACS), LAPD Police Stations

## How to Run
1. Clone the repo
```bash
   git clone https://github.com/Eman07/Bias-Aware-Crime-Hotspot-Analysis-in-Los-Angeles.git
   cd Bias-Aware-Crime-Hotspot-Analysis-in-Los-Angeles
```
2. Install dependencies
```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
```
3. Run the analysis
```bash
   python crime_hotspot_analysis.py
```

## Results
- Identified distinct crime hotspot clusters across LA neighborhoods
- Surfaced disparities in arrest rates relative to income levels and police station proximity
- Generated insights to support data-informed resource allocation strategies
