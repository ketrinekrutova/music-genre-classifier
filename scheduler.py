"""
Планировщик автоматического запуска пайплайна.
Каждые 5 минут запускает run_pipeline.py (требование ассайнмента про автоматизацию).
"""
import time
import traceback
import subprocess
import sys
import os

# Заставляем консоль Windows печатать кириллицу без ошибок кодировки
sys.stdout.reconfigure(encoding="utf-8")

INTERVAL_SECONDS = 5 * 60  # 5 минут; увеличь, если один прогон занимает больше времени
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    while True:
        print(f"\n===== Запуск пайплайна: {time.strftime('%Y-%m-%d %H:%M:%S')} =====", flush=True)
        try:
            subprocess.run(
                [sys.executable, os.path.join(BASE_DIR, "run_pipeline.py")],
                check=True,
            )
        except Exception:
            # Ошибка одного прогона не должна останавливать планировщик
            print("Пайплайн завершился с ошибкой:", flush=True)
            traceback.print_exc()

        print(f"Следующий запуск через {INTERVAL_SECONDS // 60} минут...", flush=True)
        time.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
