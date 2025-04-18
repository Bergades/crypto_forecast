# backtest.py
import matplotlib.pyplot as plt


def run_backtest_strategy(model_lstm, X_test_seq, test_df, scaler, train_df):
    y_pred_scaled = model_lstm.predict(X_test_seq)
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_test_actual = test_df['close'].values

    capital = 1.0
    prev_close = train_df['close'].iloc[-1]
    profits = []

    for i in range(len(test_df)):
        today_open = test_df['open'].iloc[i]
        today_close = test_df['close'].iloc[i]
        pred_close = y_pred[i][0]

        if pred_close > prev_close:
            profit_factor = today_close / today_open
            capital *= profit_factor
            profits.append(profit_factor - 1)
        else:
            profits.append(0)

        prev_close = today_close

    final_return_pct = (capital - 1) * 100
    print(f"Конечный капитал (при стартовом 1.0): {capital:.2f}")
    print(f"Доходность стратегии за тестовый период: {final_return_pct:.2f}%")

    return y_pred, y_test_actual


def plot_predictions(test_df, y_test_actual, y_pred, symbol):
    plt.figure(figsize=(12, 6))
    plt.plot(test_df.index, y_test_actual, label='Фактическая цена')
    plt.plot(test_df.index, y_pred.flatten(), label='Прогноз Bi-LSTM (t+1)')
    plt.title(f"Прогноз vs Фактическая цена для {symbol} на тестовом периоде")
    plt.xlabel("Дата")
    plt.ylabel("Цена закрытия")
    plt.legend()
    plt.grid(True)
    plt.show()
