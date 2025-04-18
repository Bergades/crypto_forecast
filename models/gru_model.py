# gru_model.py

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam


def train_final_lstm_and_gru(X_train_seq, y_train_seq, best_params, window_size=60):
    """
    Обучает финальные модели Bi-LSTM и Bi-GRU с параметрами из Optuna.

    Параметры:
    - X_train_seq: numpy array последовательностей для обучения, форма (n_samples, window_size, 1)
    - y_train_seq: numpy array истинных значений, форма (n_samples,)
    - best_params: dict с ключами 'units', 'dropout_rate', 'learning_rate', 'batch_size'
    - window_size: размер входного окна

    Возвращает:
    - model_lstm: обученная модель Bi-LSTM
    - model_gru: обученная модель Bi-GRU (cuDNN ускорение)
    """
    # Извлекаем лучшие гиперпараметры
    best_units   = best_params['units']
    best_dropout = best_params['dropout_rate']
    best_lr      = best_params['learning_rate']
    best_batch   = best_params['batch_size']
    epochs_final = 100

    # --- Модель A: Bi-LSTM ---
    model_lstm = Sequential()
    model_lstm.add(Bidirectional(
        LSTM(best_units, activation='tanh', recurrent_dropout=0.0),
        input_shape=(window_size, 1)
    ))
    model_lstm.add(Dropout(best_dropout))
    model_lstm.add(Dense(1))
    model_lstm.compile(
        optimizer=Adam(learning_rate=best_lr),
        loss='mean_squared_error'
    )
    model_lstm.fit(
        X_train_seq, y_train_seq,
        epochs=epochs_final,
        batch_size=best_batch,
        verbose=1
    )
    print("Финальная модель Bi-LSTM обучена.")

    # --- Модель B: Bi-GRU с поддержкой cuDNN ---
    model_gru = Sequential()
    model_gru.add(Bidirectional(
        GRU(
            best_units,
            activation='tanh',              # cuDNN талап
            recurrent_activation='sigmoid',  # cuDNN талап
            dropout=0.0,                    # cuDNN талап
            recurrent_dropout=0.0,          # cuDNN талап
            reset_after=True,               # cuDNN талап
        ),
        input_shape=(window_size, 1)
    ))
    model_gru.add(Dropout(best_dropout))
    model_gru.add(Dense(1))
    model_gru.compile(
        optimizer=Adam(learning_rate=best_lr),
        loss='mean_squared_error'
    )
    model_gru.fit(
        X_train_seq, y_train_seq,
        epochs=epochs_final,
        batch_size=best_batch,
        verbose=1
    )
    print("Модель Bi-GRU обучена.")

    return model_lstm, model_gru
