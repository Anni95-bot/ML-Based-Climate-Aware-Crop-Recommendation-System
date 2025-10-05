# ML based
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(layout="centered", page_title="ML Crop Recommender")

@st.cache_data
def load_data():
    crop_df = pd.read_csv("dataset/crop.csv")
    temp_df = pd.read_csv("dataset/temp.csv")
    rain_df = pd.read_csv("dataset/rainfall.csv")
    return crop_df, temp_df, rain_df

crop_df, temp_df, rain_df = load_data()


rf = joblib.load("models/crop_rf_model.pkl")
state_le = joblib.load("models/state_encoder.pkl")
season_le = joblib.load("models/season_encoder.pkl")
crop_le = joblib.load("models/crop_encoder.pkl")

st.title("ML Based Climate Aware Crop Recommendation System")


state = st.selectbox("Choose State", sorted(temp_df['State'].unique()))
districts = sorted(temp_df[temp_df['State'] == state]['District'].unique())
district = st.selectbox("Choose District", districts)
season = st.selectbox("Choose Season", ["Kharif", "Rabi", "Zaid"])

st.markdown("Enter soil nutrient values (N, P, K) and pH:")
col1, col2, col3, col4 = st.columns(4)
with col1: user_N = st.number_input("N (kg/ha)", value=50.0)
with col2: user_P = st.number_input("P (kg/ha)", value=30.0)
with col3: user_K = st.number_input("K (kg/ha)", value=40.0)
with col4: user_PH = st.number_input("pH", value=6.5)


temp_col_map = {"Kharif":"Temp_Kharif","Rabi":"Temp_Rabi","Zaid":"Temp_Zaid"}
rain_col_map = {"Kharif":"Rain_Kharif","Rabi":"Rain_Rabi","Zaid":"Rain_Zaid"}

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
    # Use global ranges from full crop dataset to avoid negative while making calculatios
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

        # Overall Success Rate
        success_rate = 0.6 * climate_sim + 0.4 * soil_sim
        results.append({"Crop": row['Crop'], "Success Rate (%)": round(success_rate*100,2)})

    if len(results) == 0:
        return pd.DataFrame()
    return pd.DataFrame(results).sort_values("Success Rate (%)", ascending=False)


if st.button("Recommend Crops"):
    district_temp, district_rain = get_district_climate(state, district, season)
    if district_temp is None:
        st.error("Climate data not found for this district.")
    else:
        st.write(f"District {district} — {season} Temp: **{district_temp} °C**, Rainfall: **{district_rain} mm**")

        # Filtering crop by State and  Season
        seasonal_crops = crop_df[(crop_df['State']==state) & (crop_df['Season']==season)]

        if seasonal_crops.empty:
            st.warning("No crops found for this State & Season in crop.csv.")
        else:
            # Encoding ML input for these seasonal crops
            state_enc = state_le.transform([state])[0]
            season_enc = season_le.transform([season])[0]

            X_user = pd.DataFrame([{
                'State_enc': state_enc,
                'Season_enc': season_enc,
                'Temperature': district_temp,
                'Rainfall': district_rain,
                'N': user_N, 'P': user_P, 'K': user_K, 'PH': user_PH
            }])

            probs = rf.predict_proba(X_user)[0]
            crops_all = crop_le.inverse_transform(rf.classes_)

            # taking only seasonal crops
            seasonal_indices = [i for i, c in enumerate(crops_all) if c in seasonal_crops['Crop'].values]
            if len(seasonal_indices) == 0:
                st.warning("ML predicted crops are not available for this State+Season in crop.csv. No crops to show.")
            else:
                probs_seasonal = probs[seasonal_indices]
                top_idx = np.argsort(probs_seasonal)[-3:][::-1]
                top3_crops = [crops_all[seasonal_indices[i]] for i in top_idx]

                # Computing Success Rate for crops
                top_candidates = seasonal_crops[seasonal_crops['Crop'].isin(top3_crops)]
                scores_df = compute_scores(top_candidates, district_temp, district_rain, user_N, user_P, user_K, user_PH)
                top3 = scores_df.reset_index(drop=True)
                top3.index = np.arange(1, len(top3)+1)

                st.subheader("Top Recommended Crops")
                st.table(top3)

                st.subheader("Success Rate of Recommended Crops")
                st.bar_chart(top3.set_index("Crop")["Success Rate (%)"])
