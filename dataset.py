# dataset.py
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

def create_sequences(close_scaled_full, train_size, window_size=60):
    X_seq = []
    y_seq = []
    for i in range(window_size, len(close_scaled_full)):
        X_seq.append(close_scaled_full[i - window_size:i])
        y_seq.append(close_scaled_full[i])
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    train_seq_count = train_size - window_size
    X_train_seq = X_seq[:train_seq_count]
    y_train_seq = y_seq[:train_seq_count]
    X_test_seq = X_seq[train_seq_count:]
    y_test_seq = y_seq[train_seq_count:]

    print(f"Последовательностей для обучения: {len(X_train_seq)}, для теста: {len(X_test_seq)}")

    return X_seq, y_seq, X_train_seq, y_train_seq, X_test_seq, y_test_seq

def get_time_series_split():
    return TimeSeriesSplit(n_splits=5)


