"""
Берём train/test из data/processed, делаем feature engineering (масштабирование),
обучаем модель, считаем метрики на тесте, логируем всё в MLflow и сохраняем модель.
"""
import os
import sys
import json
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")

TARGET_COL = "label"
N_ESTIMATORS = 200  # количество деревьев в случайном лесе


def main():
    train_df = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))

    feature_names = [c for c in train_df.columns if c != TARGET_COL]

    X_train, y_train = train_df[feature_names], train_df[TARGET_COL]
    X_test, y_test = test_df[feature_names], test_df[TARGET_COL]

    # Приводим все признаки к одному масштабу (среднее 0, дисперсия 1)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    os.makedirs(MODELS_DIR, exist_ok=True)

    # MLflow хранит эксперименты локально в sqlite базе (новые версии MLflow
    # больше не поддерживают простое файловое хранилище ./mlruns)
    mlflow_db_path = os.path.join(BASE_DIR, "mlflow.db")
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db_path}")
    mlflow.set_experiment("music_genre_classification")

    with mlflow.start_run():
        # Обучаем модель классификации жанров
        model = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=67)
        model.fit(X_train_scaled, y_train)

        # Оцениваем качество на отложенной тестовой выборке
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-score: {f1:.4f}")

        # Логируем параметры, метрики и саму модель в MLflow
        mlflow.log_param("n_estimators", N_ESTIMATORS)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "model")

    # Сохраняем для API предсказаний
    joblib.dump(model, os.path.join(MODELS_DIR, "model.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    with open(os.path.join(MODELS_DIR, "feature_names.json"), "w") as f:
        json.dump(feature_names, f)
    with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
        json.dump({"accuracy": accuracy, "f1_score": f1}, f)

    # Примеры строк для кнопки "заполнить случайно" в приложении
    sample_rows = test_df[feature_names].sample(n=min(30, len(test_df)), random_state=None)
    with open(os.path.join(MODELS_DIR, "sample_rows.json"), "w") as f:
        json.dump(sample_rows.to_dict(orient="records"), f)

    print(f"Модель и метрики сохранены в {MODELS_DIR}")


if __name__ == "__main__":
    main()
