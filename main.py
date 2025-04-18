# main.py

from config import symbol, timeframe, start_date
from fetch_data import load_ohlcv_data
from preprocessing import split_and_scale, create_sequences, get_time_series_split
from models.lstm_model import run_lstm_optuna
from models.gru_model import train_final_lstm_and_gru
from models.xgb_models import train_xgb_models
from backtest import run_backtest_strategy, plot_predictions
from forecast import predict_next_7_days
import optuna

# 1. Загрузка исторических данных
print(f"Загружаем данные для {symbol}, таймфрейм {timeframe} с {start_date}...")
df = load_ohlcv_data(symbol, timeframe, start_date)

# 2. Предобработка и последовательности
train_df, test_df, scaled_close, scaler = split_and_scale(df)
X_train_seq, y_train_seq, X_test_seq, y_test_seq = create_sequences(scaled_close, len(train_df))

# 3. TimeSeriesSplit и Optuna подбор
tscv = get_time_series_split()
best_params = run_lstm_optuna(X_train_seq, y_train_seq, tscv)

# 4. Финальное обучение Bi-LSTM и Bi-GRU
model_lstm, model_gru = train_final_lstm_and_gru(X_train_seq, y_train_seq, best_params)

# 5. Обучение XGBoost моделей по горизонту
models_xgb = train_xgb_models(train_df)

# 6. Backtest стратегия
y_pred, y_test_actual = run_backtest_strategy(model_lstm, X_test_seq, test_df, scaler, train_df)

# 7. График
plot_predictions(test_df, y_test_actual, y_pred, symbol)

# 8. Прогноз на 7 дней
predict_next_7_days(df, models_xgb)

