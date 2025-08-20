
import gradio as gr
import numpy as np
import pandas as pd
from pathlib import Path
import re
import joblib
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

# ---------- Paths ----------
BASE = Path(__file__).resolve().parent
CLEAN_PATH = BASE / "cleaned_listings.csv"
MODEL_PATH = BASE / "keras_model.h5"              # model salvat în H5
SCALER_Y_PATH = BASE / "scaler_y.pkl"
PPM_FINE = BASE / "baseline_ppm_by_loc_type_rooms.csv"
PPM_LOC = BASE / "baseline_ppm_by_location.csv"

# ---------- Features ----------
FEATURES = ["Surface", "Rooms", "Floor", "Location", "Appartment_type", "Price_per_m2"]

# ---------- Functions ----------
def map_location(loc):
    if pd.isna(loc): return np.nan
    loc = str(loc).lower()
    if "sectorul 1" in loc: return 1
    if "sectorul 2" in loc: return 2
    if "sectorul 3" in loc: return 3
    if "sectorul 4" in loc: return 4
    if "sectorul 5" in loc: return 5
    if "sectorul 6" in loc: return 6
    return 0  # outside Bucharest

def map_floor(floor):
    if pd.isna(floor): return np.nan
    floor = str(floor).lower()
    if "parter" in floor or "demisol" in floor: return 0
    m = re.search(r"(\d+)", floor)
    return int(m.group(1)) if m else np.nan

def rooms_to_bin(rooms):
    try:
        r = int(rooms)
    except:
        return None
    if r <= 1: return "1"
    if r == 2: return "2"
    if r == 3: return "3"
    if r == 4: return "4"
    return "5+"

df_clean = pd.read_csv(CLEAN_PATH)

# Refit scaler_X
scaler_X = StandardScaler().fit(df_clean[FEATURES].values.astype(float))

# Load model with compile=False, then compile
model = keras.models.load_model(MODEL_PATH, compile=False)
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.MeanSquaredError(),
    metrics=[keras.metrics.MeanAbsoluteError()]
)

scaler_y = joblib.load(SCALER_Y_PATH)

# Baseline €/mp
ppm_fine = pd.read_csv(PPM_FINE)   # col: Location, Appartment_type, Rooms_bin, Price_per_m2
ppm_loc = pd.read_csv(PPM_LOC)     # col: Location, Price_per_m2

def suggest_ppm(location_text, appartment_type, rooms):
    """ Sugests €/mp from baseline, with fallback. """
    loc_code = map_location(location_text)
    apt_type = 1 if str(appartment_type).strip().lower() in ["noua", "nouă", "nou"] else 0
    rbin = rooms_to_bin(rooms)

    if rbin is not None and not pd.isna(loc_code):
        m = ppm_fine[
            (ppm_fine["Location"] == loc_code) &
            (ppm_fine["Appartment_type"] == apt_type) &
            (ppm_fine["Rooms_bin"] == rbin)
        ]
        if len(m) > 0:
            return float(m["Price_per_m2"].iloc[0])

    m2 = ppm_loc[ppm_loc["Location"] == loc_code]
    if len(m2) > 0:
        return float(m2["Price_per_m2"].iloc[0])

    return float(df_clean["Price_per_m2"].median())

def compute_features(surface, rooms, floor_text, location_text, appartment_type, price_per_m2):
    loc_code = map_location(location_text)
    rooms = pd.to_numeric(rooms, errors="coerce")
    floor = map_floor(floor_text)
    apt_type = 1 if str(appartment_type).strip().lower() in ["noua", "nouă", "nou"] else 0

    x = pd.DataFrame([{
        "Surface": surface,
        "Rooms": rooms,
        "Floor": floor,
        "Location": loc_code,
        "Appartment_type": apt_type,
        "Price_per_m2": price_per_m2
    }])

    if x[FEATURES].isna().any(axis=None):
        missing = x[FEATURES].isna().sum()
        raise ValueError(f"Missing values in Features: \n{missing}")

    X_scaled = scaler_X.transform(x[FEATURES].values.astype(float))
    return X_scaled

def predict_price(surface, rooms, floor_text, location_text, appartment_type, price_per_m2):
    try:
        X_scaled = compute_features(surface, rooms, floor_text, location_text, appartment_type, price_per_m2)
        y_pred_scaled = model.predict(X_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled).ravel()[0]

        reference_total = float(price_per_m2) * float(surface) if surface and price_per_m2 else None
        if reference_total is not None:
            return f"Preț estimat (model): {y_pred:,.0f} €\n(Comparație €/mp * suprafață: {reference_total:,.0f} €)"
        return f"Preț estimat (model): {y_pred:,.0f} €"
    except Exception as e:
        return f"Eroare: {e}"

def update_default_ppm(location_text, appartment_type, rooms):
    try:
        ppm = suggest_ppm(location_text, appartment_type, rooms)
        return float(round(ppm, 0))
    except Exception:
        return None

# ---------- UI Gradio ----------
with gr.Blocks(title="Estimare Preț Apartament (Keras)") as demo:
    gr.Markdown("## 🏠 Estimare Preț Apartament (Keras + Baseline €/mp)")
    with gr.Row():
        surface = gr.Number(label="Suprafață (mp)", value=60, precision=0)
        rooms = gr.Number(label="Număr camere", value=2, precision=0)
        floor = gr.Textbox(label="Etaj (ex: Parter, 3/10)", value="Parter")
    with gr.Row():
        location = gr.Textbox(label="Locație (ex: Sectorul 3, Titan)", value="Sectorul 3")
        app_type = gr.Radio(choices=["SH", "Nouă"], value="SH", label="Tip apartament")
        ppm = gr.Number(label="€ / mp (poți ajusta)", value=0, precision=0)

    # Setează automat €/mp din baseline când se schimbă câmpurile relevante
    location.change(fn=update_default_ppm, inputs=[location, app_type, rooms], outputs=ppm)
    app_type.change(fn=update_default_ppm, inputs=[location, app_type, rooms], outputs=ppm)
    rooms.change(fn=update_default_ppm, inputs=[location, app_type, rooms], outputs=ppm)

    predict_btn = gr.Button("Calculează preț")
    output = gr.Textbox(label="Rezultat")

    predict_btn.click(
        fn=predict_price,
        inputs=[surface, rooms, floor, location, app_type, ppm],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch()
