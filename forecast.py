import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.exc import ProgrammingError
from keras.models import load_model
import ta
import datetime
import argparse

# Параметры подключения к базе данных
db_config = {
    'user': 'traider',
    'password': '123456traider',
    'host': 'localhost',
    'port': '5435',
    'database': 'db_traider'
}

# Функция для подключения к базе данных и выгрузки данных
def load_data_from_db(currency_pair, start_time, end_time=None):
    engine = create_engine(f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}")
    query = f"""
    SELECT timestamp, {currency_pair}
    FROM public.couples_relationship
    WHERE timestamp >= '{start_time}'
    """
    if end_time:
        query += f" AND timestamp <= '{end_time}'"
    query += " ORDER BY timestamp"
    
    try:
        data = pd.read_sql(query, engine)
        print(f"Loaded {len(data)} rows from the database")
        print(f"First timestamp: {data['timestamp'].iloc[0]}")
        print(f"Last timestamp: {data['timestamp'].iloc[-1]}")
        return data
    except ProgrammingError as e:
        print(f"Error: {e}")
        print("Check if the table 'public.couples_relationship' exists and the currency pair column is correct.")
        return None

# Функция для расчета индикаторов
def add_indicators(data, currency_pair):
    data['high'] = data[currency_pair] * 1.001  # предположительно на 0.1% выше текущей цены
    data['low'] = data[currency_pair] * 0.999  # предположительно на 0.1% ниже текущей цены
    data['close'] = data[currency_pair]
    
    data['SMA'] = ta.trend.sma_indicator(data['close'], window=5)
    data['EMA'] = ta.trend.ema_indicator(data['close'], window=5)
    data['RSI'] = ta.momentum.rsi(data['close'], window=14)
    data['MACD'] = ta.trend.macd(data['close'])
    data['MACD_signal'] = ta.trend.macd_signal(data['close'])
    data['MACD_diff'] = ta.trend.macd_diff(data['close'])
    data['Bollinger_high'] = ta.volatility.bollinger_hband(data['close'])
    data['Bollinger_low'] = ta.volatility.bollinger_lband(data['close'])
    data['Stochastic'] = ta.momentum.stoch(data['high'], data['low'], data['close'], window=14, smooth_window=3)
    data['ATR'] = ta.volatility.average_true_range(data['high'], data['low'], data['close'], window=3)
    
    data.dropna(inplace=True)
    return data

# Функция для выполнения прогнозирования
def forecast(currency_pair, data, model, scaler):
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data.set_index('timestamp', inplace=True)
    
    # Добавление индикаторов
    data = add_indicators(data, currency_pair)
    
    # Преобразование данных для LSTM
    X = scaler.transform(data[['SMA', 'EMA', 'RSI', 'MACD', 'MACD_signal', 'MACD_diff', 'Bollinger_high', 'Bollinger_low', 'Stochastic', 'ATR']])
    X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
    
    # Прогнозирование
    predictions = model.predict(X)
    
    # Масштабирование предсказанных значений обратно к исходному масштабу
    predicted_prices_scaled = np.zeros((predictions.shape[0], X.shape[2]))
    predicted_prices_scaled[:, 0] = predictions[:, 0]
    predicted_prices = scaler.inverse_transform(predicted_prices_scaled)[:, 0]
    
    return predicted_prices

# Основная функция
def main(currency_pair=None, start_time=None, end_time=None):
    # Если пара не указана, берем первую попавшуюся
    if not currency_pair:
        currency_pair = 'eur_usd'
    
    # Если start_time не указан, используем текущую дату и время
    if not start_time:
        start_time = datetime.datetime.now().isoformat()
    
    # Печать параметров времени
    print(f"Loading data for {currency_pair} starting from {start_time}")
    if end_time:
        print(f"Ending at {end_time}")

    data = load_data_from_db(currency_pair, start_time, end_time)
    if data is None or data.empty:
        print("No data loaded or data is empty. Exiting...")
        return
    
    # Загрузка модели и скалера
    model = load_model('lstm_model.h5')
    scaler = np.load('scaler.npy', allow_pickle=True).item()
    
    # Выполнение прогнозирования
    predicted_prices = forecast(currency_pair, data, model, scaler)
    
    # Вывод результата в консоль
    current_time = data.index[-1]
    actual_price = data[currency_pair].iloc[-1]
    predicted_price = predicted_prices[-1]
    predicted_time = current_time + datetime.timedelta(minutes=4)
    direction = 'UP' if predicted_price > actual_price else 'DOWN'
    
    result = {
        'currency': currency_pair,
        'current_time': current_time.isoformat(),
        'actual_price': actual_price,
        'predicted_price': predicted_price,
        'predicted_time': predicted_time.isoformat(),
        'direction': direction
    }
    
    print(result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Forecast currency pair prices.')
    parser.add_argument('--currency_pair', type=str, help='Currency pair to forecast')
    parser.add_argument('--start_time', type=str, help='Start time for data extraction (default: current time)')
    parser.add_argument('--end_time', type=str, help='End time for data extraction')
    args = parser.parse_args()
    main(currency_pair=args.currency_pair, start_time=args.start_time, end_time=args.end_time)
