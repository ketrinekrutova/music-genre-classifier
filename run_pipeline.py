"""
Главный скрипт всего пайплайна: данные -> модель -> деплой.
Запускает все три стадии ассайнмента одну за другой.
"""
import subprocess
import sys
import os

# Заставляем консоль Windows печатать кириллицу без ошибок кодировки
sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def run(cmd, cwd=None):
    print(f"\n>>> {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(f"Команда завершилась с ошибкой: {' '.join(cmd)}")


def main():
    python = sys.executable

    print("=== Stage 1: Data Engineering ===")
    run([python, os.path.join("code", "datasets", "data_pipeline.py")], cwd=BASE_DIR)

    print("\n=== Stage 2: Model Engineering ===")
    run([python, os.path.join("code", "models", "train_model.py")], cwd=BASE_DIR)

    print("\n=== Stage 3: Deployment ===")
    compose_file = os.path.join("code", "deployment", "docker-compose.yml")
    # Пересобираем и (пере)запускаем контейнеры api и app с новой моделью
    run(["docker-compose", "-f", compose_file, "up", "--build", "-d"], cwd=BASE_DIR)

    print("\nПайплайн успешно выполнен!")
    print("API:  http://localhost:8000/docs")
    print("App:  http://localhost:8501")


if __name__ == "__main__":
    main()
