import pandas as pd

def deep_analyze_demand(state):
    """Triggered conditionally if Risk is High. Calculates variance and deeper pattern stats."""
    df = state.get('df_raw')
    insights = state.get('insights', {})
    
    if df is not None and not df.empty:
        target_col = 'AI_Predicted_Demand_kW' if 'AI_Predicted_Demand_kW' in df.columns else 'EV Charging Demand (kW)'
        variance = float(df[target_col].var())
        insights["demand_variance"] = variance
        insights["deep_analysis_note"] = "High volatility detected via deep analysis node. Grid stabilization is urgently required."
        
    return {"insights": insights}
