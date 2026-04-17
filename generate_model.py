import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pickle
import os

def create_and_save_model():
    file_path = "data/Charging station_C__Calif.csv"
    if not os.path.exists(file_path):
        print("Data file not found.")
        return
        
    df = pd.read_csv(file_path)
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='mixed')
    df = df.sort_values('Datetime').reset_index(drop=True)
    
    target_col = 'EV Charging Demand (kW)'
    
    df['Hour'] = df['Datetime'].dt.hour
    df['DayOfWeek'] = df['Datetime'].dt.dayofweek
    df['Demand_Lag_1'] = df[target_col].shift(1)
    df['Demand_Lag_2'] = df[target_col].shift(2)
    df['Rolling_Avg_3h'] = df[target_col].rolling(window=3).mean().shift(1)
    
    df = df.dropna().reset_index(drop=True)
    
    features = [
        'Hour',
        'DayOfWeek',
        'Demand_Lag_1',
        'Demand_Lag_2',
        'Rolling_Avg_3h',
        'Electricity Price ($/kWh)',
        'Grid Stability Index',
        'Number of EVs Charging'
    ]
    
    X = df[features].astype(float)
    y = df[target_col]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_scaled, y)
    
    os.makedirs('models', exist_ok=True)
    
    with open('models/ev_demand_timeseries.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
        
    print("Model and scaler trained and saved successfully.")

if __name__ == "__main__":
    create_and_save_model()
