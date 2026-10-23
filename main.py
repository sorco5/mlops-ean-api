# API FastAPI que recibe los datos de un estudiante y consulta el endpoint MLOPS_EAN de Databricks
import os
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Credenciales: se inyectan como variables de entorno en Azure (nunca escritas en el código)
DATABRICKS_URL = os.getenv("DATABRICKS_URL")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")

app = FastAPI(title="API MLOps EAN", description="Predicción de diabtes")

# Permite que el frontend (index.html) llame a la API desde otro dominio
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# Campos que espera el modelo (deben coincidir con la firma registrada en MLflow)
class Estudiante(BaseModel):
    embarazos: int = Field(..., ge=0, le=20)
    glucosa: float = Field(..., ge=0, le=300)
    presion_diastolica: float = Field(..., ge=0, le=200)
    pliegue_cutaneo: float = Field(..., ge=0, le=100)
    insulina: float = Field(..., ge=0, le=1000)
    imc: float = Field(..., ge=0, le=100)
    funcion_pedigri_diabetes: float = Field(..., ge=0,     le=3)
    edad: int = Field(..., ge=0, le=120)
    mediciones_faltantes: int = Field(..., ge=0, le=10)
    interaccion_glucosa_imc: float = Field(..., ge=0, le=20000)
    interaccion_edad_imc: float = Field(..., ge=0, le=12000)


@app.get("/health")
def health():
    # Verifica que la API está viva y que tiene configuradas las credenciales
    url_ok = bool(DATABRICKS_URL and DATABRICKS_URL.strip().startswith("https://"))
    token_ok = bool(DATABRICKS_TOKEN and DATABRICKS_TOKEN.strip())
    return {
        "status": "ok",
        "databricks_url": url_ok,
        "databricks_token": token_ok,
        "databricks_configurado": url_ok and token_ok,
    }


@app.post("/predecir")
def predecir(est: Estudiante):
    if not (DATABRICKS_URL and DATABRICKS_TOKEN):
        raise HTTPException(status_code=500, detail="Faltan DATABRICKS_URL o DATABRICKS_TOKEN")

    # Databricks exige el formato dataframe_records
    payload = {"dataframe_records": [est.model_dump()]}

    try:
        r = requests.post(
            DATABRICKS_URL,
            headers={"Authorization": f"Bearer {DATABRICKS_TOKEN}"},
            json=payload,
            timeout=120,  # margen para el arranque en frío del endpoint
        )
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Databricks tardó demasiado (arranque en frío). Intente de nuevo.")

    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Error de Databricks: {r.text}")

    pred = int(r.json()["predictions"][0])  # 1 = aprueba, 0 = reprueba
    return {
        "prediccion": pred,
        "resultado": "Con diabetes ❌" if pred == 1 else "Sin diabetes ✅",
        "datos_recibidos": est.model_dump(),
    }