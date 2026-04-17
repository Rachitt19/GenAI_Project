import pandas as pd

def analyze_demand(state):
    df = state.get('df_raw')
    insights = {
        "peak_hours": [],
        "high_load_periods": [],
        "risk_level": "unknown",
        "message": "No data processed."
    }
    
    if df is None or df.empty:
        return {"insights": insights}
        
    try:
        # We assume the dataframe has been processed by preprocess_data in app.py
        target_col = 'AI_Predicted_Demand_kW' if 'AI_Predicted_Demand_kW' in df.columns else 'EV Charging Demand (kW)'
        
        # Make sure Datetime exists or can be created
        if 'Datetime' not in df.columns:
            if 'Date' in df.columns and 'Time' in df.columns:
                df['Datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str), format='mixed')
            else:
                return {"insights": insights}

        # Peak hours based on average demand
        hourly_avg = df.groupby(df['Datetime'].dt.hour)[target_col].mean()
        peak_hours = hourly_avg.nlargest(3).index.tolist()
        
        # High-load periods
        threshold = df[target_col].quantile(0.9)
        high_load_df = df[df[target_col] >= threshold]
        high_load_periods = high_load_df['Datetime'].dt.strftime('%Y-%m-%d %H:%M').tolist()[:5]
        
        risk_level = "High" if len(high_load_df) > len(df) * 0.15 else "Moderate"
        
        insights = {
            "peak_hours": peak_hours,
            "high_load_periods": high_load_periods,
            "risk_level": risk_level,
            "max_demand": float(df[target_col].max()),
            "avg_demand": float(df[target_col].mean()),
            "status": "Success"
        }
    except Exception as e:
        insights["message"] = f"Error during analysis: {str(e)}"
        
    return {"insights": insights}
