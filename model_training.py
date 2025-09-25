import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import json
import joblib

# Load raw data
raw_df = pd.read_csv("/Users/cosmindanaita/PycharmProjects/AI3/CosminD/housing_predictions//raw_listings.csv")

# Split off 10% as "real-world" test data before any cleaning
real_test_df = raw_df.sample(frac=0.1, random_state=42)
train_df = raw_df.drop(real_test_df.index)
real_test_df.to_csv("/Users/cosmindanaita/PycharmProjects/AI3/CosminD/housing_predictions/test_real_world.csv", index=False)

# Work only on train data for now
df = train_df.copy()
# Backup original values
df["Price_raw"] = df["Price"]
df["Size_raw"] = df["Size"]
df["Location_raw"] = df["Location"]
df["Rooms_raw"] = df["Rooms"]
df["Floor_raw"] = df["Floor"]
df["Heating_raw"] = df["Heating"]
df["Elevator_raw"] = df["Elevator"]
df["Appartment_type_raw"] = df["Appartment_type"]
df["Seller_raw"] = df["Seller"]

def map_location(loc):
    if pd.isna(loc):
        return np.nan
    loc = str(loc).lower()
    if "sectorul 1" in loc:
        return 1
    if "sectorul 2" in loc:
        return 2
    if "sectorul 3" in loc:
        return 3
    if "sectorul 4" in loc:
        return 4
    if "sectorul 5" in loc:
        return 5
    if "sectorul 6" in loc:
        return 6
    return 0  # exterior

df["Location"] = df["Location"].apply(map_location)
print(df[["Location"]].head(5))

df["Rooms"] = pd.to_numeric(df["Rooms"], errors="coerce")
df = df[df["Rooms"] <= 5]
print(df[["Rooms"]].head(5))

# Clean Floor
def map_floor(floor):
    if pd.isna(floor):
        return np.nan
    floor = str(floor).lower()
    if "parter" in floor or "demisol" in floor:
        return 0
    match = pd.Series(floor).str.extract(r"(\d+)").iloc[0, 0]
    return pd.to_numeric(match, errors="coerce")

df["Floor"] = df["Floor"].apply(map_floor)
print(df[["Floor"]].head(5))

df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
print(df[["Year"]].head(5))

df["Heating"] = df["Heating"].apply(
    lambda x: 1 if str(x).strip().lower() in ["da", "centrală pe gaz", "centralizată"] else 0
)
print(df[["Heating"]].head(5))

df["Elevator"] = df["Elevator"].apply(
    lambda x: 1 if str(x).strip().lower() == "da" else 0
)
print(df[["Elevator"]].head(5))

df["Appartment_type"] = df["Appartment_type"].apply(
    lambda x: 1 if "nouă" in str(x).lower() else 0
)
print(df[["Appartment_type"]].head(5))


df["Seller"] = df["Seller"].apply(
    lambda x: 1 if "agen" in str(x).lower() else 0  # 1 = agentie, 0 = proprietar
)
print(df[["Seller"]].head(5))


df["Size"] = df["Size"].astype(str).str.extract(r"(\d{2,3})", expand=False)
df["Size"] = pd.to_numeric(df["Size"], errors="coerce")
df.rename(columns={"Size": "Surface"}, inplace=True)
print(df[["Surface"]].head(5))

# Remove Surface outliers using IQR
q1_s, q3_s = df["Surface"].quantile([0.25, 0.75])
iqr_s = q3_s - q1_s
lower_bound_s = q1_s - 1.5 * iqr_s
upper_bound_s = q3_s + 1.5 * iqr_s
df = df[(df["Surface"] >= lower_bound_s) & (df["Surface"] <= upper_bound_s)]
# df = df[df["Surface"] < 101]

df["Price"] = (
    df["Price"]
    .astype(str)
    .str.replace("\xa0", "", regex=False)  # eliminates non-breaking space
    .str.replace("€", "", regex=False)     # eliminates €
    .str.replace(",", "", regex=False)     # eliminates ,
)
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
print(df[["Price"]].head(5))

# Remove Price outliers using IQR
q1_p, q3_p = df["Price"].quantile([0.25, 0.75])
iqr_p = q3_p - q1_p
lower_bound_p = q1_p - 1.5 * iqr_p
upper_bound_p = q3_p + 1.5 * iqr_p
df = df[(df["Price"] >= lower_bound_p) & (df["Price"] <= upper_bound_p)]
# df = df[df["Price"] < 200000]

# Addins
df["Price_per_m2"] = df["Price"] / df["Surface"]
df["Rooms_Surface"] = df["Rooms"] * df["Surface"]

# Drop rows with missing values
features = ["Surface", "Rooms", "Floor", "Location", "Appartment_type", "Price_per_m2"]
target = "Price"
print("[INFO] Original size:", df.shape)
print("[INFO] Invalid columns before dropna:")
print(df[features + [target]].isna().sum())
df = df.dropna(subset=features + [target])
print("[INFO] Size after drop:", df.shape)
print("[INFO] Invalid columns after dropna:")
print(df[features + [target]].isna().sum())



# Save cleaned data with original values
cleaned_path = "/Users/cosmindanaita/PycharmProjects/AI3/CosminD/housing_predictions/cleaned_listings.csv"
df.to_csv(cleaned_path, index=False)

# --- Exploratory Plots ---
for feature in features + [target]:
    plt.figure(figsize=(10, 5))
    sns.histplot(df[feature], kde=True, bins=50)
    plt.title(f"Distribuția pentru {feature}")
    plt.xlabel(feature)
    plt.ylabel("Număr de apartamente")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 2))
    sns.boxplot(x=df[feature])
    plt.title(f"Boxplot pentru {feature}")
    plt.tight_layout()
    plt.show()

# --- Comparative Plots ---
sns.pairplot(df[["Price", "Surface", "Rooms", "Floor"]])
plt.suptitle("Corelații între variabile", y=1.02)
plt.show()

# Define features and target
X = df[features]
y = df["Price"].values.reshape(-1, 1)

# Normalize features
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

# Normalize target
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Build the model
model = keras.Sequential([
    keras.Input(shape=(X_train.shape[1],)),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(64, activation="relu"),
    layers.Dense(1)
])

model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# Train
early_stop = EarlyStopping(monitor="val_loss", patience=50, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=500,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

bins = [0, 1, 2, 3, 4, 10]  # camerele 1,2,3,4,5+ (ajustează cum vrei)
labels = ["1", "2", "3", "4", "5+"]
df["Rooms_bin"] = pd.cut(df["Rooms"], bins=bins, labels=labels, right=True, include_lowest=True)

group_cols = ["Location", "Appartment_type", "Rooms_bin"]
baseline_ppm = (
    df.dropna(subset=["Price_per_m2"] + group_cols)
      .groupby(group_cols)["Price_per_m2"]
      .median()
      .reset_index()
)

# fallback general pe Location (fără app_type/rooms), dacă lipsește o combinație fină
baseline_ppm_location = (
    df.dropna(subset=["Price_per_m2", "Location"])
      .groupby(["Location"])["Price_per_m2"]
      .median()
      .reset_index()
)

baseline_ppm.to_csv("baseline_ppm_by_loc_type_rooms.csv", index=False)
baseline_ppm_location.to_csv("baseline_ppm_by_location.csv", index=False)

# salvează modelul și scaler_y (dacă n-ai făcut-o)
model.save("keras_model.h5")

joblib.dump(scaler_y, "scaler_y.pkl")

# Predict and evaluate
y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_orig = scaler_y.inverse_transform(y_test)

mae = mean_absolute_error(y_test_orig, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred))
r2 = r2_score(y_test_orig, y_pred)

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R^2 Score: {r2:.4f}")


# Predict on real-world test set
real_test_clean = real_test_df.copy()

# Apply same cleaning as above
real_test_clean["Price"] = real_test_clean["Price"].astype(str).str.replace("\xa0", "", regex=False).str.replace("€", "", regex=False).str.replace(",", "", regex=False)
real_test_clean["Price"] = pd.to_numeric(real_test_clean["Price"], errors="coerce")
real_test_clean["Size"] = real_test_clean["Size"].astype(str).str.extract(r"(\d{2,3})", expand=False)
real_test_clean["Size"] = pd.to_numeric(real_test_clean["Size"], errors="coerce")
real_test_clean.rename(columns={"Size": "Surface"}, inplace=True)
real_test_clean["Location"] = real_test_clean["Location"].apply(map_location)
real_test_clean["Rooms"] = pd.to_numeric(real_test_clean["Rooms"], errors="coerce")
real_test_clean = real_test_clean[real_test_clean["Rooms"] <= 5]
real_test_clean["Floor"] = real_test_clean["Floor"].apply(map_floor)
real_test_clean["Appartment_type"] = real_test_clean["Appartment_type"].apply(lambda x: 1 if "nouă" in str(x).lower() else 0)
real_test_clean["Price_per_m2"] = real_test_clean["Price"] / real_test_clean["Surface"]
real_test_clean = real_test_clean.dropna(subset=features + [target])

# Predict
X_real = real_test_clean[features]
y_real = real_test_clean["Price"].values.reshape(-1, 1)
X_real_scaled = scaler_X.transform(X_real)
y_real_pred_scaled = model.predict(X_real_scaled)
y_real_pred = scaler_y.inverse_transform(y_real_pred_scaled)

mae_real = mean_absolute_error(y_real, y_real_pred)
rmse_real = np.sqrt(mean_squared_error(y_real, y_real_pred))
r2_real = r2_score(y_real, y_real_pred)

print("--- Evaluation on Real-World Test Set ---")
print(f"MAE: {mae_real:.2f}")
print(f"RMSE: {rmse_real:.2f}")
print(f"R^2 Score: {r2_real:.4f}")


