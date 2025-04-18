# lstm_model.py — подбор, обучение и оценка Bi-LSTM с Optuna
import optuna
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


def run_lstm_optuna(X_train_seq, y_train_seq, tscv, window_size=60, n_trials=30, timeout=600):
    """
    Подбор гиперпараметров Bi-LSTM через Optuna.
    Возвращает лучшие параметры.
    """
    def objective_lstm(trial):
        units = trial.suggest_int('units', 20, 100)
        dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        epochs = 50

        mse_scores = []
        for train_index, val_index in tscv.split(X_train_seq):
            X_tr, X_val = X_train_seq[train_index], X_train_seq[val_index]
            y_tr, y_val = y_train_seq[train_index], y_train_seq[val_index]

            model = Sequential([
                Bidirectional(LSTM(units, activation='tanh', recurrent_activation='sigmoid'),
                              input_shape=(window_size, 1)),
                Dropout(dropout_rate),
                Dense(1)
            ])
            model.compile(
                optimizer=Adam(learning_rate=learning_rate),
                loss='mean_squared_error',
                metrics=['mean_absolute_error', 'mean_absolute_percentage_error']
            )

            es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            model.fit(
                X_tr, y_tr,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[es],
                verbose=0
            )

            y_val_pred = model.predict(X_val, verbose=0)
            mse_scores.append(mean_squared_error(y_val, y_val_pred))

            trial.report(mse_scores[-1], len(mse_scores))
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return float(np.mean(mse_scores))

    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    )
    study.optimize(objective_lstm, n_trials=n_trials, timeout=timeout)

    print("Наилучшие параметры для Bi-LSTM:", study.best_params)
    return study.best_params


def build_and_train_lstm(X_train_seq, y_train_seq, best_params, window_size=60):
    """
    Строит и обучает финальную модель Bi-LSTM на всем тренировочном наборе.
    Возвращает обученную модель.
    """
    model = Sequential([
        Bidirectional(LSTM(best_params['units'], activation='tanh', recurrent_activation='sigmoid'),
                       input_shape=(window_size, 1)),
        Dropout(best_params['dropout_rate']),
        Dense(1)
    ])
    model.compile(
        optimizer=Adam(learning_rate=best_params['learning_rate']),
        loss='mean_squared_error',
        metrics=['mean_absolute_error', 'mean_absolute_percentage_error']
    )

    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(
        X_train_seq, y_train_seq,
        validation_split=0.1,
        epochs=100,
        batch_size=best_params['batch_size'],
        callbacks=[es],
        verbose=1
    )
    return model


def evaluate_model(model, X_test_seq, y_test_seq):
    """
    Оценивает модель на тестовом наборе и печатает MSE, MAE, MAPE.
    """
    y_pred = model.predict(X_test_seq)
    mse = mean_squared_error(y_test_seq, y_pred)
    mae = mean_absolute_error(y_test_seq, y_pred)
    mape = mean_absolute_percentage_error(y_test_seq, y_pred)
    print(f"Test MSE : {mse:.6f}")
    print(f"Test MAE : {mae:.6f}")
    print(f"Test MAPE: {mape:.2%}")
    return {'mse': mse, 'mae': mae, 'mape': mape}


# Пример использования:
# if __name__ == "__main__":
#     from sklearn.model_selection import TimeSeriesSplit
#     # Предположим, X_train_seq и y_train_seq подготовлены извне
#     tscv = TimeSeriesSplit(n_splits=5)
#     best_params = run_lstm_optuna(X_train_seq, y_train_seq, tscv)
#     final_model = build_and_train_lstm(X_train_seq, y_train_seq, best_params)
#     evaluate_model(final_model, X_test_seq, y_test_seq)
