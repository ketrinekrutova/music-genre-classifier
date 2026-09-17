# Классификация музыкальных жанров — MLOps пайплайн

Автоматизированный пайплайн: обработка данных → обучение модели → деплой модели в виде API и веб-приложения.

## Структура репозитория

```
code/
  datasets/data_pipeline.py     - Stage 1: загрузка, очистка, разбиение данных
  models/train_model.py         - Stage 2: обучение и оценка модели, логирование в MLflow
  deployment/
    api/                        - FastAPI сервис с моделью
    app/                        - Streamlit приложение
    docker-compose.yml          - поднимает api и app в отдельных контейнерах
data/
  raw/                          - исходный датасет (csv)
  processed/                    - train.csv / test.csv (создаются автоматически)
models/                         - обученная модель, скейлер, метрики (создаются автоматически)
run_pipeline.py                 - один полный прогон всего пайплайна
scheduler.py                    - запускает run_pipeline.py каждые 5 минут
requirements.txt                - зависимости для data/model стадий
```

## Датасет

Используется [GTZAN Genre Collection](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)
(файл с готовыми аудио-признаками, например `features_30_sec.csv`). Файл нужно положить в `data/raw/`.

## Как запустить

1. Установить зависимости:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Положить csv файл с данными в `data/raw/`.
3. Убедиться, что установлен и запущен Docker Desktop.
4. Запустить пайплайн один раз:
   ```
   python run_pipeline.py
   ```
5. Чтобы пайплайн запускался автоматически каждые 5 минут, оставить работать:
   ```
   python scheduler.py
   ```

## Доступ к сервисам

- Веб-приложение: http://localhost:8501
- API (документация Swagger): http://localhost:8000/docs
- MLflow эксперименты: `mlflow ui --backend-store-uri mlruns` → http://localhost:5000

## Как это работает

1. **Data Engineering** (`code/datasets/data_pipeline.py`) — читает csv из `data/raw`, убирает пропуски и выбросы, делит на train/test, сохраняет в `data/processed`.
2. **Model Engineering** (`code/models/train_model.py`) — масштабирует признаки, обучает `RandomForestClassifier`, считает accuracy/F1 на тесте, логирует всё в MLflow, сохраняет модель в `models/`.
3. **Deployment** (`code/deployment/`) — API загружает модель и отдаёт предсказания, приложение собирает форму ввода по списку признаков из API и показывает предсказанный жанр.
4. **Автоматизация** (`scheduler.py`) — каждые 5 минут заново прогоняет весь пайплайн и пересобирает Docker-контейнеры с обновлённой моделью.
