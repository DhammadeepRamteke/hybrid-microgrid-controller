import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import os

# FIX: Force the app to use the current folder as the working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# --- 1. SETUP & CONFIGURATION ---
st.set_page_config(page_title="Smart Microgrid Controller", layout="wide")

st.title("⚡ AI-Powered Smart Microgrid Controller")
st.markdown("""
**Internship Project: Hybrid Microgrid Energy Management System**
This dashboard simulates energy generation and demand. It uses historical data patterns 
combined with real-time user inputs to optimize battery usage.
""")

# --- 2. LOAD ARTIFACTS ---
@st.cache_resource
def load_data_and_models():
    # 1. Load the Dataset (To get the correct feature structure)
    df = pd.read_csv('processed_dataset.csv')
    
    # 2. Load Models & Scaler
    scaler = joblib.load('scaler.joblib')
    solar_model = joblib.load('model_solar.joblib')
    wind_model = joblib.load('model_wind.joblib')
    load_model = joblib.load('model_load.joblib')
    
    return df, scaler, solar_model, wind_model, load_model

try:
    df_ref, scaler, solar_model, wind_model, load_model = load_data_and_models()
    st.success("System Online: Models & Data Pipeline Loaded")
except FileNotFoundError as e:
    st.error(f"Error loading files: {e}")
    st.stop()

# --- 3. DEFINE THE 21 FEATURES (Exact Order from Training) ---
FEATURE_COLS = [
    'solar_irradiance', 'wind_speed', 'temperature', 'humidity', 'pressure',
    'grid_frequency', 'grid_voltage', 'grid_exchange', 'battery_soc',
    'battery_charge', 'battery_discharge', 'hour', 'day_of_week', 'month',
    'day_of_year', 'load_demand_lag1', 'solar_irradiance_lag1',
    'wind_speed_lag1', 'wind_speed_roll_3h', 'solar_irradiance_roll_3h',
    'load_roll_3h'
]

# --- 4. SIDEBAR: USER INPUTS ---
st.sidebar.header("Simulation Parameters")

# Time Simulation
selected_month = st.sidebar.selectbox("Month", list(range(1, 13)), index=10) # Default Nov
selected_hour = st.sidebar.slider("Time of Day (Hour)", 0, 23, 12)

st.sidebar.subheader("Weather Conditions")
temp_input = st.sidebar.slider("Temperature (°C)", -10.0, 50.0, 25.0)
wind_input = st.sidebar.slider("Wind Speed (m/s)", 0.0, 30.0, 5.0)
irradiance_input = st.sidebar.slider("Solar Irradiance (W/m²)", 0.0, 1200.0, 500.0)

# --- 5. PREDICTION LOGIC ---
def get_prediction_input(month, hour, temp, wind, irr):
    match = df_ref[(df_ref['month'] == month) & (df_ref['hour'] == hour)]
    
    if match.empty:
        base_row = df_ref[df_ref['month'] == month].mean().to_frame().T
    else:
        base_row = match.iloc[0:1].copy()
    
    base_row['temperature'] = temp
    base_row['wind_speed'] = wind
    base_row['solar_irradiance'] = irr
    
    input_data = base_row[FEATURE_COLS]
    return input_data

if st.sidebar.button("Run Simulation"):
    # Prepare Input
    input_df = get_prediction_input(selected_month, selected_hour, temp_input, wind_input, irradiance_input)
    
    # Scale Input
    try:
        input_scaled = scaler.transform(input_df)
    except ValueError:
        input_scaled = input_df

    # Predict
    pred_solar = max(0, solar_model.predict(input_scaled)[0])
    pred_wind = max(0, wind_model.predict(input_scaled)[0])
    pred_load = max(0, load_model.predict(input_scaled)[0])

    # --- 6. SMART CONTROLLER LOGIC ---
    total_gen = pred_solar + pred_wind
    net_energy = total_gen - pred_load
    
    if net_energy >= 0:
        status = "SURPLUS: Charging Battery / Exporting"
        color = "green"
    else:
        status = "DEFICIT: Discharging Battery / Importing"
        color = "red"

    # --- 7. VISUALIZATION ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Solar Output", f"{pred_solar:.2f} kW")
    col2.metric("Wind Output", f"{pred_wind:.2f} kW")
    col3.metric("Load Demand", f"{pred_load:.2f} kW", delta_color="inverse")
    
    st.divider()
    st.subheader(f"Grid Status: :{color}[{status}]")
    
    # Bar Chart
    fig = go.Figure(data=[
        go.Bar(name='Generation', x=['Solar', 'Wind'], y=[pred_solar, pred_wind], marker_color='#2ecc71'),
        go.Bar(name='Demand', x=['Load'], y=[pred_load], marker_color='#e74c3c')
    ])
    fig.update_layout(barmode='group', title="Power Balance (kW)", height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Gauge
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = net_energy,
        title = {'text': "Net Energy (kW)"},
        gauge = {'axis': {'range': [-1000, 1000]},
                 'bar': {'color': "black"},
                 'steps': [
                     {'range': [-1000, 0], 'color': "#ffcccc"},
                     {'range': [0, 1000], 'color': "#ccffcc"}]}))
    st.plotly_chart(fig_gauge, use_container_width=True)

else:
    st.info("Adjust parameters on the sidebar and click 'Run Simulation'")
