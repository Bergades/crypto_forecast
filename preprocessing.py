# preprocessing.py

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit

def split_and_scale(df):
    # Разделение на обучающую и тестовую выборки
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    print(f"Обучающих точек: {len(train_df)}, Тестовых точек: {len(test_df)}")

    # Масштабирование (нормализация) значений закрытия для нейронных сетей
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_close_values = train_df['close'].values.reshape(-1, 1)
    scaler.fit(train_close_values)

    # Применяем масштабирование ко всему ряду закрытия
    close_scaled_full = scaler.transform(df['close'].values.reshape(-1, 1))

    return train_df, test_df, close_scaled_full, scaler

def create_sequences(close_scaled_full, train_size, window_size=60):
    # Параметры последовательностей
    sequence_data = close_scaled_full
    X_seq = []
    y_seq = []
    for i in range(window_size, len(sequence_data)):
        X_seq.append(sequence_data[i - window_size:i])
        y_seq.append(sequence_data[i])
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    # Разделение последовательностей на train и test по исходному train_size
    train_seq_count = train_size - window_size
    X_train_seq = X_seq[:train_seq_count]
    y_train_seq = y_seq[:train_seq_count]
    X_test_seq = X_seq[train_seq_count:]
    y_test_seq = y_seq[train_seq_count:]

    print(f"Последовательностей для обучения: {len(X_train_seq)}, для теста: {len(X_test_seq)}")
    return X_train_seq, y_train_seq, X_test_seq, y_test_seq

def get_time_series_split():
    return TimeSeriesSplit(n_splits=5)