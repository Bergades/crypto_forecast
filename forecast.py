# forecast.py
from datetime import timedelta


def predict_next_7_days(df, models_xgb, lags=30):
    # Формирование признаков из последних lags дней для прогноза
    last_known_prices = df['close'].values[-lags:]
    X_last = last_known_prices.reshape(1, -1)

    future_preds = []
    last_date = df.index[-1]
    for h in range(1, 8):
        pred_price = models_xgb[h].predict(X_last)[0]
        future_date = last_date + timedelta(days=h)
        future_preds.append((future_date.date(), pred_price))

    print("Прогноз цены закрытия на следующие 7 дней:")
    for date, price in future_preds:
        print(f"{date}: {price:.2f}")

