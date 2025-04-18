# config.py

import optuna
import numpy as np

symbol = input("Введите торговую пару (например, ETH/USDT): ").strip()
if symbol == "":
    symbol = "ETH/USDT"

timeframe = '1d'
start_date = '2021-01-01T00:00:00Z'  # дата начала загрузки исторических данных

print(f"Загружаем данные для {symbol}, таймфрейм {timeframe} с {start_date}...")
