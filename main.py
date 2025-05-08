from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

# Inisialisasi FastAPI
app = FastAPI(title="API Prediksi Pengeluaran CPS")

# Load model dan scaler
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Skema input
class Transaction(BaseModel):
    Tanggal: str
    Customer: str
    Nama_Kapal: str
    Nominal_yang_Dibayarkan: float
    DPP: float
    PPM: float

# Fungsi untuk membuat fitur baru
def create_features(data: Transaction):
    # Buat DataFrame dari input
    df = pd.DataFrame([{
        "Tanggal": pd.to_datetime(data.Tanggal),
        "Customer": data.Customer,
        "Nama Kapal": data.Nama_Kapal,
        "Nominal yang Dibayarkan": data.Nominal_yang_Dibayarkan,
        "DPP": data.DPP,
        "PPM": data.PPM
    }])

    # Feature Engineering
    df['Biaya Administrasi'] = df['Nominal yang Dibayarkan'] - (df['DPP'] + df['PPM'])
    df['Bulan Tahun'] = df['Tanggal'].dt.to_period('M').astype(str)

    # Kolom yang akan digunakan untuk prediksi
    feature_columns = [
        "Nominal yang Dibayarkan", "DPP", "PPM", "Biaya Administrasi"
    ]

    # Normalisasi data
    df_scaled = scaler.transform(df[feature_columns])

    return df_scaled

@app.get("/")
def read_root():
    return {"message": "API Prediksi Pengeluaran CPS Sedang Berjalan..."}

# Endpoint prediksi
@app.post("/predict")
def predict_cluster(data: Transaction):
    processed = create_features(data)
    prediction = model.predict(processed)[0]
    
    # Mapping nilai cluster
    cluster_mapping = {0: "Rendah", 1: "Sedang", 2: "Tinggi"}
    cluster_label = cluster_mapping.get(int(prediction), "Tidak Diketahui")
    
    return {
        "Customer": data.Customer,
        "Nama Kapal": data.Nama_Kapal,
        "KMeans_Label": int(prediction),
        "Prediksi": cluster_label
    }