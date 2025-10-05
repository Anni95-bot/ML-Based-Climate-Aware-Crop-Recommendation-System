# Rule based 
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(layout="centered", page_title="Rule-based Climate Aware Crop Recommendation System")

@st.cache_data
def load_data():
    crop_df = pd.read_csv("dataset/crop.csv")
    temp_df = pd.read_csv("dataset/temp.csv")
    rain_df = pd.read_csv("dataset/rainfall.csv")
    return crop_df, temp_df, rain_df

crop_df, temp_df, rain_df = load_data()

st.title("Rule-based Climate Aware Crop Recommendation System")

state = st.selectbox("Choose State", sorted(temp_df['State'].unique()))
districts = sorted(temp_df[temp_df['State'] == state]['District'].unique())
district = st.selectbox("Choose District", districts)
season = st.selectbox("Choose Season", ["Kharif", "Rabi", "Zaid"])

st.markdown("Enter soil nutrient values (N, P, K) and pH:")
col1, col2, col3, col4 = st.columns(4)
with col1:
    user_N = st.number_input("N (kg/ha)", value=50.0, step=1.0)
with col2:
    user_P = st.number_input("P (kg/ha)", value=30.0, step=1.0)
with col3:
    user_K = st.number_input("K (kg/ha)", value=40.0, step=1.0)
with col4:
    user_PH = st.number_input("pH", value=6.5, format="%.2f")

temp_col_map = {"Kharif":"Temp_Kharif", "Rabi":"Temp_Rabi", "Zaid":"Temp_Zaid"}
rain_col_map = {"Kharif":"Rain_Kharif", "Rabi":"Rain_Rabi", "Zaid":"Rain_Zaid"}

def get_district_climate(state, district, season):
    tcol = temp_col_map[season]
    rcol = rain_col_map[season]
    temp_row = temp_df[(temp_df['State'] == state) & (temp_df['District'] == district)]
    rain_row = rain_df[(rain_df['State'] == state) & (rain_df['District'] == district)]
    if temp_row.empty or rain_row.empty:
        return None, None
    return float(temp_row.iloc[0][tcol]), float(rain_row.iloc[0][rcol])

def compute_scores(crops, district_temp, district_rain, user_N, user_P, user_K, user_PH):
    eps = 1e-6
    # Use global ranges to avoid negative success rate
    temp_range = max(crop_df['Temperature'].max() - crop_df['Temperature'].min(), eps)
    rain_range = max(crop_df['Rainfall'].max() - crop_df['Rainfall'].min(), eps)
    N_range = max(crop_df['N'].max() - crop_df['N'].min(), eps)
    P_range = max(crop_df['P'].max() - crop_df['P'].min(), eps)
    K_range = max(crop_df['K'].max() - crop_df['K'].min(), eps)
    PH_range = max(crop_df['PH'].max() - crop_df['PH'].min(), eps)

    results = []
    for _, row in crops.iterrows():
        # Climate score
        temp_diff = min(abs(district_temp - float(row['Temperature'])) / temp_range, 1.0)
        rain_diff = min(abs(district_rain - float(row['Rainfall'])) / rain_range, 1.0)
        climate_sim = 1.0 - 0.5 * (temp_diff + rain_diff)
        climate_sim = max(0.0, climate_sim)

        # Soil score
        n_diff = min(abs(user_N - float(row['N'])) / N_range, 1.0)
        p_diff = min(abs(user_P - float(row['P'])) / P_range, 1.0)
        k_diff = min(abs(user_K - float(row['K'])) / K_range, 1.0)
        ph_diff = min(abs(user_PH - float(row['PH'])) / PH_range, 1.0)
        soil_sim = 1.0 - 0.25 * (n_diff + p_diff + k_diff + ph_diff)
        soil_sim = max(0.0, soil_sim)

        # Success rate
        success_rate = 0.6 * climate_sim + 0.4 * soil_sim

        results.append({"Crop": row['Crop'], "Success Rate (%)": round(success_rate*100, 2)})

    return pd.DataFrame(results).sort_values("Success Rate (%)", ascending=False)

if st.button("Recommend Crops"):
    district_temp, district_rain = get_district_climate(state, district, season)
    if district_temp is None:
        st.error("Climate data not found for the selected district")
    else:
        st.write(f"District {district} — {season} temperature: **{district_temp} °C**, rainfall: **{district_rain} mm**")
        # filter crop by state & season
        candidates = crop_df[(crop_df['State'] == state) & (crop_df['Season'] == season)].copy()
        if candidates.empty:
            st.warning("No crops found for this State & Season in crop.csv.")
        else:
            scores_df = compute_scores(candidates, district_temp, district_rain, user_N, user_P, user_K, user_PH)
            top3 = scores_df.head(3).reset_index(drop=True)
            top3.index = np.arange(1, len(top3)+1)
            st.subheader("Top 3 Recommended Crops")
            st.table(top3)
            # Bar chart
            st.subheader("Success Rate of Top 3 Crops")
            st.bar_chart(top3.set_index("Crop")["Success Rate (%)"])
