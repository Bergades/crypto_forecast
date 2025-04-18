# fetch_data.py

import ccxt
import pandas as pd
from datetime import datetime

def load_ohlcv_data(symbol, timeframe, start_date):
    # Инициализация объекта биржи Coinbase
    exchange = ccxt.coinbase({'apiKey': 'f5234f93-3315-4019-be1e-2079526cc326'})
    exchange.load_markets()  # загрузить маркеты (может потребоваться для некоторых бирж)

    # Получение метки времени начала в формате миллисекунд
    since_ts = exchange.parse8601(start_date)

    # Загрузка исторических свечей OHLCV
    all_ohlcv = []
    while True:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ts, limit=200)
        if not ohlcv:
            break
        all_ohlcv.extend(ohlcv)
        # Обновляем since_ts для следующей итерации как время последней полученной свечи + 1 мс
        last_ts = ohlcv[-1][0]
        since_ts = last_ts + 1
        # Если получили менее 200 записей, выходим (достигнут конец)
        if len(ohlcv) < 200:
            break

    # Преобразование в DataFrame
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    # Конвертация метки времени в дату
    df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('Date', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']]  # упорядочим колонки
    print(f"Загружено {len(df)} дневных свечей.")
    print(df.tail(3))
    return df
