"""
Загружаем сырые данные из data/raw, чистим их (пропуски, выбросы)
и сохраняем train/test выборки в data/processed.
"""
import os
import sys
import glob
import pandas as pd
from sklearn.model_selection import train_test_split

# Заставляем консоль Windows печатать кириллицу без ошибок кодировки
sys.stdout.reconfigure(encoding="utf-8")

# Пути к папкам с данными (относительно этого файла)
BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

# Типичные названия колонки с жанром (целевая переменная)
POSSIBLE_TARGET_NAMES = ["label", "genre", "class", "target"]
# Служебные колонки, которые не являются признаками для модели
ID_COLUMNS = ["filename", "file_name", "id"]

TARGET_COL_OUT = "label"  # под этим именем сохраняем целевую колонку в processed-файлах
TEST_SIZE = 0.2
RANDOM_STATE = 67


def find_raw_file():
    """Находит csv файл с сырыми данными в папке data/raw"""
    csv_files = glob.glob(os.path.join(RAW_DIR, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"В папке {RAW_DIR} не найден csv файл с данными")
    return csv_files[0]


def find_target_column(df):
    """Ищет колонку с жанром по типичным названиям"""
    for name in POSSIBLE_TARGET_NAMES:
        for col in df.columns:
            if col.lower() == name:
                return col
    raise ValueError(
        "Не удалось найти целевую колонку (жанр). "
        f"Ожидались имена: {POSSIBLE_TARGET_NAMES}. Есть колонки: {list(df.columns)}"
    )


def clean_data(df, target_col):
    """Очистка данных: убираем служебные колонки, пропуски и выбросы"""
    # Убираем колонки, которые не нужны модели (например, имя файла)
    drop_cols = [c for c in df.columns if c.lower() in ID_COLUMNS]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Оставляем только числовые признаки + целевую колонку
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    df = df[numeric_cols + [target_col]].copy()
    df = df.rename(columns={target_col: TARGET_COL_OUT})

    # Убираем полные дубликаты строк
    df = df.drop_duplicates()

    # Заполняем пропуски медианой по каждому признаку
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # Обрезка выбросов по методу IQR (значение, а не строка целиком — иначе при ~50
    # признаках потеряли бы почти весь датасет)
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        df[col] = df[col].clip(lower, upper)

    return df


def split_and_save(df):
    """Разбивает данные на train/test и сохраняет в data/processed"""
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[TARGET_COL_OUT],
    )

    train_path = os.path.join(PROCESSED_DIR, "train.csv")
    test_path = os.path.join(PROCESSED_DIR, "test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Train: {train_df.shape} -> {train_path}")
    print(f"Test:  {test_df.shape} -> {test_path}")


def main():
    raw_path = find_raw_file()
    print(f"Загружаем данные из файла: {raw_path}")
    df = pd.read_csv(raw_path)

    target_col = find_target_column(df)
    print(f"Целевая колонка (жанр): {target_col}")

    df_clean = clean_data(df, target_col)
    print(f"После очистки осталось строк: {len(df_clean)} (было {len(df)})")

    split_and_save(df_clean)


if __name__ == "__main__":
    main()
