import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from io import BytesIO
from utils import apply_terminal_theme, print_terminal_log
import sys
import os
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from agent.run_agent import run_planning_agent
except Exception as e:
    run_planning_agent = None
    print(f"Warning: agent module failed to load. {e}")

st.set_page_config(page_title="NEURAL GRID | EV FORECAST", layout="wide")
apply_terminal_theme()

@st.cache_resource
def load_model():
    try:
        # Load the model from the current directory
        return joblib.load('models/ev_demand_timeseries.pkl')
    except Exception:
        return None

predictor = load_model()

def preprocess_data(df_raw):
    """
    Transforms raw station CSV format into model-ready features.
    """
    df = df_raw.copy()
    try:
        # 1. Temporal Features
        df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='mixed')
        df['Hour'] = df['Datetime'].dt.hour
        df['DayOfWeek'] = df['Datetime'].dt.dayofweek
        
        # 2. Time-Series Memory (Lags & Rolling)
        # Using the target column from the uploaded file to create memory
        target_col = 'EV Charging Demand (kW)'
        df['Demand_Lag_1'] = df[target_col].shift(1)
        df['Demand_Lag_2'] = df[target_col].shift(2)
        df['Rolling_Avg_3h'] = df[target_col].rolling(window=3).mean().shift(1)
        
        # 3. Rename columns to match the trained model's feature list
        mapping = {
            'Electricity Price ($/kWh)': 'Electricity Price ($/kWh)',
            'Grid Stability Index': 'Grid Stability Index',
            'Number of EVs Charging': 'Number of EVs Charging'
        }
        
        df = df.fillna(method='bfill').fillna(method='ffill')
        return df
    except Exception as e:
        st.error(f"Processing Error: {str(e)}")
        return None

st.title("NEURAL GRID: EV DEMAND FORECASTING")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["Manual Prediction", "Raw File Batch Processing", "AI Infrastructure Planner"])

with tab1:
    col_input, col_viz = st.columns([1, 2])
    with col_input:
        h = st.slider("Hour", 0, 23, 12)
        d = st.slider("Day (0=Mon, 6=Sun)", 0, 6, 0)
        l1 = st.number_input("Demand Lag-1 (kW)", value=0.1500, format="%.4f")
        l2 = st.number_input("Demand Lag-2 (kW)", value=0.1450, format="%.4f")
        r3 = st.number_input("Rolling 3h (kW)", value=0.1480, format="%.4f")
        pr = st.number_input("Price ($/kWh)", value=0.1200, format="%.4f")
        stb = st.number_input("Stability Index", value=1.0000, format="%.4f")
        evc = st.number_input("EV Count", value=5)

        if st.button("RUN INFERENCE"):
            if predictor:
                # Feature order must match training
                features = [h, d, l1, l2, r3, pr, stb, evc]
                prediction = predictor.predict([features])[0]
                st.metric("PREDICTED LOAD", f"{prediction:.4f} kW")
            else:
                st.error("Model Error: Serialization file not detected in root.")

    with col_viz:
        x = np.linspace(0, 23, 100)
        y = 0.15 + 0.1 * np.sin((x - 6) * np.pi / 12)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, fill='tozeroy', line=dict(color='#00f2ff'), name='Load Profile'))
        fig.add_trace(go.Scatter(x=[h], y=[0.15 + 0.1 * np.sin((h - 6) * np.pi / 12)], mode='markers', marker=dict(color='red', size=12)))
        fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Process Raw Station Data")
    st.write("Upload a raw station file (e.g., Charging station_C__Calif.csv)")
    uploaded_file = st.file_uploader("Select CSV or Excel", type=["csv", "xlsx"])
    
    if uploaded_file and predictor:
        raw_data = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        print_terminal_log("Raw stream detected. Commencing feature engineering...")
        
        if st.button("EXECUTE BATCH INFRASTRUCTURE ANALYSIS"):
            processed_df = preprocess_data(raw_data)
            
            if processed_df is not None:
                # The exact list of features the model was trained on
                model_features = ['Hour', 'DayOfWeek', 'Demand_Lag_1', 'Demand_Lag_2', 'Rolling_Avg_3h', 
                                  'Electricity Price ($/kWh)', 'Grid Stability Index', 'Number of EVs Charging']
                
                # Generate Predictions
                X = processed_df[model_features]
                processed_df['AI_Predicted_Demand_kW'] = predictor.predict(X)
                
                print_terminal_log("Inference complete. Generating output stream...")
                st.dataframe(processed_df[['Date', 'Time', 'EV Charging Demand (kW)', 'AI_Predicted_Demand_kW']].head(10))
                
                # Download Option
                csv_buffer = BytesIO()
                processed_df.to_csv(csv_buffer, index=False)
                st.download_button("DOWNLOAD PROCESSED CSV WITH PREDICTIONS", csv_buffer.getvalue(), "Inference_Report.csv", "text/csv")
                
                # Save to session state for agent planning
                st.session_state['processed_df'] = processed_df

st.markdown("---")
print_terminal_log("System Idle. Awaiting data packet...")

with tab3:
    st.subheader("Agentic EV Infrastructure Planner")
    st.write("Reason over predicted demand & retrieve planning guidelines using Open-Source LangGraph + RAG pipeline.")
    
    # We display insights, RAG retrieval, LLM reasoning, and Final plan
    
    if st.button("RUN AGENTIC PLANNER"):
        if 'processed_df' in st.session_state and st.session_state['processed_df'] is not None:
            df_to_use = st.session_state['processed_df']
            
            with st.spinner("Agent is reasoning... (This might take a moment using local/fallback model)"):
                try:
                    if run_planning_agent is not None:
                        result = run_planning_agent(df_to_use)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("### Demand Insights")
                            st.json(result.get("insights", {}))
                            st.markdown("### AI Reasoning Process")
                            st.json(result.get("reasoning", {}))
                        
                        with col2:
                            st.markdown("### Retrieved Knowledge Base")
                            st.write(result.get("retrieved_knowledge", []))
                            st.markdown("### What-If Simulation")
                            st.json(result.get("simulated_impact", {}))
                        
                        st.markdown(f"### Final Planning Recommendations (Loops: {result.get('iteration_count')})")
                        st.json(result.get("final_plan", {}))
                    else:
                        st.error("Agent module is missing or failed to import. Check paths and dependencies.")
                        
                except Exception as e:
                    st.error(f"Agent Execution Error: {str(e)}")
        else:
            st.warning("Please upload and run batch inference in 'Raw File Batch Processing' tab first to generate predicted demand before running the agent.")