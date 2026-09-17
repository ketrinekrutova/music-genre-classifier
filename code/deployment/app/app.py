"""
Streamlit-приложение с двумя способами определить жанр: загрузить аудиофайл
(признаки считаются через librosa) или ввести признаки вручную.
Общается с API по HTTP.
"""
import os
import tempfile
from collections import Counter

import altair as alt
import numpy as np
import pandas as pd
import requests
import streamlit as st
import librosa

# Внутри docker-compose сети API доступен по имени сервиса "api"
API_URL = os.environ.get("API_URL", "http://localhost:8000")

# Модель обучена на кусках по 3 секунды при 22050 Гц — режем загруженный файл так же
SAMPLE_RATE = 22050
SEGMENT_SECONDS = 3
SEGMENT_SAMPLES = SAMPLE_RATE * SEGMENT_SECONDS

# Палитра ярких градиентов: зелёный/бирюза, жёлтый/оранжевый, коралл/розовый, синий/сиреневый
GRADIENTS = [
    "linear-gradient(135deg, #5B7FFF 0%, #A855F7 100%)",   # синий/сиреневый
    "linear-gradient(135deg, #FF5831 0%, #FF6FA8 100%)",   # коралл/розовый
    "linear-gradient(135deg, #2DD4BF 0%, #16A34A 100%)",   # зелёный/бирюза
    "linear-gradient(135deg, #FFC93C 0%, #FF8A3D 100%)",   # жёлтый/оранжевый
]
# Сплошные версии тех же цветов для графика голосов
PALETTE = ["#7C6CFF", "#FF5831", "#2DD4BF", "#FFC93C"]

# Русские названия жанров + эмодзи + градиент карточки (модель отвечает на английском)
GENRES = {}
for i, (key, name_ru, emoji) in enumerate([
    ("blues", "Блюз", "🎷"), ("classical", "Классика", "🎻"), ("country", "Кантри", "🤠"),
    ("disco", "Диско", "🪩"), ("hiphop", "Хип-хоп", "🎤"), ("jazz", "Джаз", "🎺"),
    ("metal", "Метал", "🤘"), ("pop", "Поп", "🎧"), ("reggae", "Регги", "🌴"), ("rock", "Рок", "🎸"),
]):
    GENRES[key] = (name_ru, emoji, GRADIENTS[i % len(GRADIENTS)])


def genre_ru(genre: str) -> str:
    return GENRES.get(genre.lower(), (genre, "🎵", GRADIENTS[0]))[0]


st.set_page_config(page_title="Music Genre Classifier", page_icon="🎵", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@600;700;800&family=Poppins:wght@400;500;600&display=swap');

html, body { font-family: 'Poppins', sans-serif; overflow-x: hidden; }
h1, h2, h3, h4, h5, h6, .hero-title, .genre-name { font-family: 'Sora', sans-serif; }

/* Декоративные размытые пятна на фоне — придают глубину тёмному шоколадному фону */
.bg-blob-1 {
    position: fixed; top: -120px; left: -100px; width: 380px; height: 380px;
    background: radial-gradient(circle, #2DD4BF 0%, transparent 70%);
    filter: blur(60px); opacity: 0.30; z-index: -1; pointer-events: none;
}
.bg-blob-2 {
    position: fixed; top: -100px; right: -120px; width: 420px; height: 420px;
    background: radial-gradient(circle, #FF5831 0%, transparent 70%);
    filter: blur(70px); opacity: 0.28; z-index: -1; pointer-events: none;
}

/* Яркие акцентные кнопки-капсулы (коралл по умолчанию) */
.stButton > button {
    background: #FF5831;
    color: #FFF8F0;
    border-radius: 999px;
    font-weight: 600;
    font-family: 'Sora', sans-serif;
    padding: 0.55rem 1.5rem;
    border: none;
    width: 100%;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 20px rgba(255, 88, 49, 0.35);
    color: #FFF8F0;
}
.stButton > button:disabled { background: #4A3826; color: #9C8A78; }

/* Кнопка "Заполнить случайно" — другого цвета (жёлтый), чтобы отличаться от "Определить жанр" */
.st-key-btn_random .stButton > button { background: #FFD93B; color: #10241F; }
.st-key-btn_random .stButton > button:hover { box-shadow: 0 8px 20px rgba(255, 217, 59, 0.35); }

/* Кнопка "Upload" внутри поля загрузки файла — крупнее и более круглая */
section[data-testid="stFileUploaderDropzone"] button[data-testid="stBaseButton-secondary"] {
    border-radius: 999px;
    padding: 0.8rem 2.2rem;
    font-size: 1.05rem;
}

.hero-title { font-size: 3.1rem; font-weight: 800; color: #FFF8F0; margin-bottom: 0; }
.hero-subtitle { color: #C9B8A8; margin-top: 0.3rem; margin-bottom: 1.8rem; font-size: 1.05rem; }

/* Карточка с итоговым жанром — градиент зависит от жанра */
.genre-card {
    border-radius: 20px;
    padding: 1.5rem;
    text-align: center;
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.35);
}
.genre-emoji { font-size: 2.6rem; line-height: 1; }
.genre-name { font-size: 1.5rem; font-weight: 800; color: white; letter-spacing: 1px; margin-top: .2rem; }
.genre-extra { color: rgba(255,255,255,0.9); margin-top: .4rem; font-size: 0.85rem; }

/* Поле загрузки: пунктирная обводка на всю ширину */
div[data-testid="stFileUploader"] {
    border: 2px dashed #FF5831;
    border-radius: 20px;
    padding: 2rem;
    background: #2E2115;
}
div[data-testid="stFileUploader"] section {
    border: none !important;
    background: transparent !important;
    padding: 0 !important;
}
section[data-testid="stFileUploaderDropzone"] {
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    min-height: 160px;
    gap: 0.6rem;
    text-align: center;
}
/* Подпись "200MB per file..." — под кнопкой, мельче и приглушённым цветом */
div[data-testid="stFileUploaderDropzoneInstructions"] {
    margin-top: 10px;
}
div[data-testid="stFileUploaderDropzoneInstructions"] span {
    font-size: 0.75rem;
    color: #9C8A78;
}

/* Карточка-панель для графика голосов */
.st-key-chart_card {
    background: #2E2115;
    border-radius: 20px;
    padding: 1.25rem 1.5rem;
}

/* Растягиваем строку колонок на полную высоту, иначе sticky не за что "цепляться" */
.st-key-manual_row div[data-testid="stHorizontalBlock"] { align-items: stretch !important; }
div[data-testid="stLayoutWrapper"]:has(> .st-key-actions_sidebar) {
    height: 100% !important;
    align-items: flex-start !important;
}

/* Боковая панель действий: не двигается при прокрутке списка признаков */
.st-key-actions_sidebar {
    position: sticky !important;
    top: 1rem;
    align-self: flex-start !important;
    background: #2E2115;
    border-radius: 18px;
    padding: 1.25rem;
    height: fit-content !important;
    flex: none !important;
}
.status-text {
    color: #C9B8A8;
    font-size: 0.9rem;
    text-align: center;
    margin: 0.75rem 0;
}
</style>
<div class="bg-blob-1"></div>
<div class="bg-blob-2"></div>
""", unsafe_allow_html=True)

st.markdown('<div class="hero-title">Классификация музыкальных жанров</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Загрузи трек или введи признаки вручную — модель определит жанр</div>', unsafe_allow_html=True)


@st.cache_data(ttl=60)
def get_feature_names():
    """Спрашивает у API, какие признаки нужны модели, чтобы построить форму"""
    response = requests.get(f"{API_URL}/features", timeout=10)
    response.raise_for_status()
    return response.json()["feature_names"]


try:
    feature_names = get_feature_names()
except Exception:
    st.error("Не удалось подключиться к API. Проверь, что контейнер api запущен.")
    st.stop()


def predict_genre(features: dict) -> str:
    """Отправляет признаки в API и возвращает предсказанный жанр (на английском, как в модели)"""
    response = requests.post(f"{API_URL}/predict", json={"features": features}, timeout=30)
    response.raise_for_status()
    return response.json()["genre"]


def show_genre_card(genre: str, extra: str = ""):
    """Карточка с итоговым жанром на русском — цвет градиента зависит от жанра"""
    name_ru, emoji, gradient = GENRES.get(genre.lower(), (genre, "🎵", GRADIENTS[0]))
    extra_html = f'<div class="genre-extra">{extra}</div>' if extra else ""
    st.markdown(f"""
    <div class="genre-card" style="background: {gradient};">
        <div class="genre-emoji">{emoji}</div>
        <div class="genre-name">{name_ru}</div>
        {extra_html}
    </div>
    """, unsafe_allow_html=True)


def show_vote_chart(genres_ru_list):
    """Цветной график голосов по кусочкам трека (закруглённые бруски, палитра градиентов)"""
    counts = pd.Series(Counter(genres_ru_list)).sort_values(ascending=False).reset_index()
    counts.columns = ["Жанр", "Голосов"]
    chart = (
        alt.Chart(counts)
        .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
        .encode(
            x=alt.X("Жанр", sort="-y", axis=alt.Axis(labelColor="#F5EFE6", labelAngle=0, title=None)),
            y=alt.Y("Голосов", axis=alt.Axis(labelColor="#F5EFE6", title=None)),
            color=alt.Color("Жанр", legend=None, scale=alt.Scale(range=PALETTE)),
        )
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False, domainColor="#4A3826")
        .properties(height=220, background="transparent")
    )
    st.altair_chart(chart, use_container_width=True)


def extract_features(y, sr):
    """Считает те же аудио-признаки, на которых обучалась модель (librosa)"""
    features = {"length": len(y)}

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features["chroma_stft_mean"], features["chroma_stft_var"] = float(chroma.mean()), float(chroma.var())

    rms = librosa.feature.rms(y=y)
    features["rms_mean"], features["rms_var"] = float(rms.mean()), float(rms.var())

    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    features["spectral_centroid_mean"] = float(spec_cent.mean())
    features["spectral_centroid_var"] = float(spec_cent.var())

    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features["spectral_bandwidth_mean"] = float(spec_bw.mean())
    features["spectral_bandwidth_var"] = float(spec_bw.var())

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features["rolloff_mean"], features["rolloff_var"] = float(rolloff.mean()), float(rolloff.var())

    zcr = librosa.feature.zero_crossing_rate(y)
    features["zero_crossing_rate_mean"] = float(zcr.mean())
    features["zero_crossing_rate_var"] = float(zcr.var())

    # Разделяем сигнал на гармоническую и перкуссионную составляющие
    harmony, perceptr = librosa.effects.hpss(y)
    features["harmony_mean"], features["harmony_var"] = float(harmony.mean()), float(harmony.var())
    features["perceptr_mean"], features["perceptr_var"] = float(perceptr.mean()), float(perceptr.var())

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    features["tempo"] = float(np.asarray(tempo).flatten()[0])

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for i in range(20):
        features[f"mfcc{i + 1}_mean"] = float(mfcc[i].mean())
        features[f"mfcc{i + 1}_var"] = float(mfcc[i].var())

    return features


def split_into_segments(y, sr):
    """Приводит трек к нужной частоте дискретизации и режет на куски по 3 секунды"""
    if sr != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
    n_segments = len(y) // SEGMENT_SAMPLES
    return [y[i * SEGMENT_SAMPLES:(i + 1) * SEGMENT_SAMPLES] for i in range(n_segments)]


tab_upload, tab_manual = st.tabs(["Загрузить аудиофайл", "Ввести признаки вручную"])

with tab_upload:
    # Поле загрузки на всю ширину
    uploaded_file = st.file_uploader("Аудиофайл", type=["wav", "mp3"], label_visibility="collapsed")

    if uploaded_file is not None:
        st.audio(uploaded_file)

    _, center_col, _ = st.columns([1, 1.2, 1])
    with center_col:
        predict_clicked = st.button(
            "Определить жанр по файлу",
            type="primary",
            use_container_width=True,
            disabled=uploaded_file is None,
        )

    if predict_clicked:
        status = st.empty()
        status.markdown('<div class="status-text">Модель обрабатывает файл...</div>', unsafe_allow_html=True)

        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        try:
            y, sr = librosa.load(tmp_path, sr=None, mono=True)
            segments = split_into_segments(y, sr)

            if not segments:
                status.empty()
                st.error("Файл слишком короткий, нужно минимум 3 секунды.")
            else:
                genres_en = [predict_genre(extract_features(seg, SAMPLE_RATE)) for seg in segments]
                final_genre_en, votes = Counter(genres_en).most_common(1)[0]
                status.empty()

                result_col1, result_col2 = st.columns([1, 1])
                with result_col1:
                    show_genre_card(final_genre_en, extra=f"{votes} из {len(genres_en)} кусков трека")
                with result_col2:
                    chart_card = st.container(key="chart_card")
                    with chart_card:
                        st.caption(f"Голоса по кусочкам трека (по {SEGMENT_SECONDS} сек)")
                        genres_ru = [genre_ru(g) for g in genres_en]
                        show_vote_chart(genres_ru)
        finally:
            os.remove(tmp_path)

with tab_manual:
    for name in feature_names:
        if f"feat_{name}" not in st.session_state:
            st.session_state[f"feat_{name}"] = 0.0

    # Колонка с кнопками идёт в коде раньше полей ввода: Streamlit не даёт менять
    # session_state виджета после того, как он уже отрисован в этом прогоне.
    # Визуально порядок [поля, действия] всё равно задаёт st.columns() ниже.
    manual_row = st.container(key="manual_row")
    with manual_row:
        fields_col, actions_col = st.columns([5, 1.4])

        with actions_col:
            sidebar = st.container(key="actions_sidebar")
            with sidebar:
                random_wrap = st.container(key="btn_random")
                with random_wrap:
                    if st.button("Заполнить случайно"):
                        try:
                            sample = requests.get(f"{API_URL}/sample", timeout=10).json()["features"]
                            for name, value in sample.items():
                                st.session_state[f"feat_{name}"] = value
                            st.rerun()
                        except Exception as e:
                            st.error(f"Не удалось получить пример: {e}")

                if st.button("Определить жанр", type="primary"):
                    status = st.empty()
                    status.markdown('<div class="status-text">Модель обрабатывает...</div>', unsafe_allow_html=True)
                    try:
                        # Берём текущие значения полей из session_state — сами поля
                        # ещё не созданы в этом прогоне (они ниже по коду)
                        current_features = {name: st.session_state.get(f"feat_{name}", 0.0) for name in feature_names}
                        genre = predict_genre(current_features)
                        status.empty()
                        show_genre_card(genre)
                    except Exception as e:
                        status.empty()
                        st.error(f"Ошибка API: {e}")

        with fields_col:
            columns = st.columns(5)
            for i, name in enumerate(feature_names):
                with columns[i % 5]:
                    st.number_input(name, format="%.3f", key=f"feat_{name}")
