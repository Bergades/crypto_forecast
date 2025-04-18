# xgb_model.py

from xgboost import XGBRegressor
import numpy as np

lags = 30  # количество лагов для признаков XGBoost

def train_xgb_models(train_df):
    train_prices = train_df['close'].values
    models_xgb = {}

    for h in range(1, 8):
        X_train_h = []
        y_train_h = []
        # Формируем датасет: для каждого индекса i, где i >= lags-1 и i+h < len(train_prices),
        # берем признаки = цены [i-(lags-1) ... i] и цель = цена на i+h
        for i in range(lags - 1, len(train_prices) - h):
            past_window = train_prices[i - (lags - 1): i + 1]  # lags значений с i-(lags-1) по i включительно
            target_value = train_prices[i + h]                 # значение через h дней
            X_train_h.append(past_window)
            y_train_h.append(target_value)
        X_train_h = np.array(X_train_h)
        y_train_h = np.array(y_train_h)

        # Инициализируем и обучаем модель XGBoost
        model = XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        model.fit(X_train_h, y_train_h, verbose=False)
        models_xgb[h] = model
        print(f"XGBoost модель для горизонта {h} обучена. Число обучающих примеров: {len(X_train_h)}")

    return models_xgb
