# train_ml_model.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load crop dataset
df = pd.read_csv("dataset/crop.csv")

# Encode categorical features
state_le = LabelEncoder()
season_le = LabelEncoder()
crop_le = LabelEncoder()

df['State_enc'] = state_le.fit_transform(df['State'])
df['Season_enc'] = season_le.fit_transform(df['Season'])
df['Crop_enc'] = crop_le.fit_transform(df['Crop'])

# Save encoders for later use
joblib.dump(state_le, "state_encoder.pkl")
joblib.dump(season_le, "season_encoder.pkl")
joblib.dump(crop_le, "crop_encoder.pkl")

# Features and target
X = df[['State_enc', 'Season_enc', 'Temperature', 'Rainfall', 'N', 'P', 'K', 'PH']]
y = df['Crop_enc']

# Train Random Forest
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X, y)

# Save model
joblib.dump(rf, "crop_rf_model.pkl")

print("ML model trained and saved successfully!")
