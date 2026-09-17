"""
FastAPI сервис: принимает числовые признаки трека и возвращает предсказанный жанр.
Модель, скейлер и список признаков подключаются через volume из папки models/.
"""
import os
import json
import random
from typing import Dict

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODELS_DIR = "/app/models"

app = FastAPI(title="Music Genre Classifier API")

model = joblib.load(os.path.join(MODELS_DIR, "model.pkl"))
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
with open(os.path.join(MODELS_DIR, "feature_names.json")) as f:
    FEATURE_NAMES = json.load(f)
with open(os.path.join(MODELS_DIR, "sample_rows.json")) as f:
    SAMPLE_ROWS = json.load(f)


class PredictRequest(BaseModel):
    features: Dict[str, float]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/features")
def get_features():
    # для формы ввода
    return {"feature_names": FEATURE_NAMES}


@app.get("/sample")
def get_sample():
    # для кнопки "заполнить случайно"
    return {"features": random.choice(SAMPLE_ROWS)}


@app.post("/predict")
def predict(request: PredictRequest):
    # Проверка, что переданы все нужные признаки
    missing = [name for name in FEATURE_NAMES if name not in request.features]
    if missing:
        raise HTTPException(status_code=400, detail=f"Не хватает признаков: {missing}")

    # Сборка признаков в том порядке, в котором обучалась модель
    x = np.array([[request.features[name] for name in FEATURE_NAMES]])
    x_scaled = scaler.transform(x)

    genre = model.predict(x_scaled)[0]
    return {"genre": str(genre)}
