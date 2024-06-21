import pandas as pd
import ta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Загрузка данных
data = pd.read_csv('data.csv')

# Преобразование столбца timestamp в datetime
data['timestamp'] = pd.to_datetime(data['timestamp'])

# Установка столбца timestamp в качестве индекса
data.set_index('timestamp', inplace=True)

# Предполагаем, что у нас есть столбцы 'high', 'low' и 'close'
data['high'] = data['eur_usd'] * 1.001  # предположительно на 0.1% выше текущей цены
data['low'] = data['eur_usd'] * 0.999  # предположительно на 0.1% ниже текущей цены
data['close'] = data['eur_usd']

# Добавление индикаторов
data['SMA'] = ta.trend.sma_indicator(data['eur_usd'], window=5)
data['EMA'] = ta.trend.ema_indicator(data['eur_usd'], window=5)
data['RSI'] = ta.momentum.rsi(data['eur_usd'], window=14)
data['MACD'] = ta.trend.macd(data['eur_usd'])
data['MACD_signal'] = ta.trend.macd_signal(data['eur_usd'])
data['MACD_diff'] = ta.trend.macd_diff(data['eur_usd'])
data['Bollinger_high'] = ta.volatility.bollinger_hband(data['eur_usd'])
data['Bollinger_low'] = ta.volatility.bollinger_lband(data['eur_usd'])
data['Stochastic'] = ta.momentum.stoch(data['high'], data['low'], data['close'], window=14, smooth_window=3)
data['ATR'] = ta.volatility.average_true_range(data['high'], data['low'], data['close'], window=14)

# Удаление строк с NaN значениями
data.dropna(inplace=True)

# Определение признаков и целевой переменной
features = data[['SMA', 'EMA', 'RSI', 'MACD', 'MACD_signal', 'MACD_diff', 'Bollinger_high', 'Bollinger_low', 'Stochastic', 'ATR']]
target = data['eur_usd']

# Нормализация данных
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, shuffle=False)

# Сохранение данных
np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)
np.save('scaler.npy', scaler)
