# Model Artifacts
This directory houses the trained model files required for deployment.

## Version 1.0 (Milestone 1)
* `ev_demand_timeseries.pkl`: Random Forest model trained on synchronized station data.
* `scaler.pkl`: StandardScaler object used for input normalization.

## Model Logic
The model utilizes a 3-hour rolling average and lag-1 demand features to achieve an 89.16% accuracy. These artifacts are loaded dynamically by the Streamlit application in `src/app.py`.