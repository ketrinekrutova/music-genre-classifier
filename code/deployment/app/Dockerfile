FROM python:3.11-slim

WORKDIR /app

# ffmpeg нужен librosa, чтобы уметь читать mp3-файлы
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY .streamlit/ .streamlit/

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
