from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError
import pickle
import pandas as pd

app = FastAPI(title="API Prediksi Pengeluaran CPS")

# Load model dan scaler
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except Exception as e:
    raise RuntimeError(f"Gagal memuat model atau scaler: {e}")

# Skema input
class Transaction(BaseModel):
    Tanggal: str
    Pelanggan: str
    Nama_Kapal: str
    Nominal_yang_Dibayarkan: float
    DPP: float
    PPM: float

# Fungsi membuat fitur
def create_features(data: Transaction):
    try:
        df = pd.DataFrame([{
            "Tanggal": pd.to_datetime(data.Tanggal),
            "Pelanggan": data.Pelanggan,
            "Nama Kapal": data.Nama_Kapal,
            "Nominal yang Dibayarkan": data.Nominal_yang_Dibayarkan,
            "DPP": data.DPP,
            "PPM": data.PPM
        }])

        df['Biaya Administrasi'] = df['Nominal yang Dibayarkan'] - (df['DPP'] + df['PPM'])
        df['Bulan Tahun'] = df['Tanggal'].dt.to_period('M').astype(str)

        feature_columns = [
            "Nominal yang Dibayarkan", "DPP", "PPM", "Biaya Administrasi"
        ]

        df_scaled = scaler.transform(df[feature_columns])
        return df_scaled
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error dalam memproses data: {e}")

# Root
@app.get("/")
def read_root():
    return {"message": "API Prediksi Pengeluaran CPS Sedang Berjalan..."}

# Endpoint prediksi
@app.post("/predict")
def predict_cluster(data: Transaction):
    try:
        processed = create_features(data)
        prediction = model.predict(processed)[0]
        cluster_mapping = {0: "Rendah", 1: "Sedang", 2: "Tinggi"}
        cluster_label = cluster_mapping.get(int(prediction), "Tidak Diketahui")

        return {
            "Pelanggan": data.Pelanggan,
            "Nama Kapal": data.Nama_Kapal,
            "KMeans_Label": int(prediction),
            "Prediksi": cluster_label
        }

    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan saat prediksi: {e}")

# Custom handler untuk kesalahan JSON
@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": exc.body},
    )
