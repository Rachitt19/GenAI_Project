import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

print("Loading data...")
df = pd.read_csv('data/Charging station_C__Calif.csv')

print("Preprocessing...")
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='mixed')
df['Hour'] = df['Datetime'].dt.hour
df['DayOfWeek'] = df['Datetime'].dt.dayofweek
target_col = 'EV Charging Demand (kW)'
df['Demand_Lag_1'] = df[target_col].shift(1)
df['Demand_Lag_2'] = df[target_col].shift(2)
df['Rolling_Avg_3h'] = df[target_col].rolling(window=3).mean().shift(1)

df = df.fillna(method='bfill').fillna(method='ffill')

features = ['Hour', 'DayOfWeek', 'Demand_Lag_1', 'Demand_Lag_2', 'Rolling_Avg_3h', 
            'Electricity Price ($/kWh)', 'Grid Stability Index', 'Number of EVs Charging']

X = df[features]
y = df[target_col]

print("Training Quick RF Model...")
model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
model.fit(X, y)

os.makedirs('models', exist_ok=True)
os.makedirs('src/models', exist_ok=True)

# Save to both root models and src/models to avoid Path issues depending on where they run it.
joblib.dump(model, 'models/ev_demand_timeseries.pkl')
joblib.dump(model, 'src/models/ev_demand_timeseries.pkl')
print("Model created and saved to models/ev_demand_timeseries.pkl and src/models/ev_demand_timeseries.pkl")
