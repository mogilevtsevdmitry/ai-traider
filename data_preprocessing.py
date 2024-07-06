# data_preprocessing.py

import pandas as pd
import ta

def add_indicators(data, currency_pair):
    data['SMA'] = ta.trend.sma_indicator(data[currency_pair], window=14)
    data['EMA'] = ta.trend.ema_indicator(data[currency_pair], window=14)
    data['RSI'] = ta.momentum.rsi(data[currency_pair], window=14)
    data['MACD'] = ta.trend.macd(data[currency_pair])
    data['MACD_signal'] = ta.trend.macd_signal(data[currency_pair])
    data['MACD_diff'] = data['MACD'] - data['MACD_signal']
    data['Bollinger_high'] = ta.volatility.bollinger_hband(data[currency_pair], window=14)
    data['Bollinger_low'] = ta.volatility.bollinger_lband(data[currency_pair], window=14)
    data['Stochastic'] = ta.momentum.stoch(data[currency_pair], data[currency_pair], data[currency_pair], window=14, smooth_window=3)
    data['ATR'] = ta.volatility.average_true_range(data[currency_pair], data[currency_pair], data[currency_pair], window=14)
    data['ADX'] = ta.trend.adx(data[currency_pair], data[currency_pair], data[currency_pair], window=14)
    data['CCI'] = ta.trend.cci(data[currency_pair], data[currency_pair], data[currency_pair], window=14)
    data['ROC'] = ta.momentum.roc(data[currency_pair], window=14)
    data['WilliamsR'] = ta.momentum.williams_r(data[currency_pair], data[currency_pair], data[currency_pair], lbp=14)

    # Проверка на NaN значения в индикаторах
    if data.isnull().values.any():
        print("Indicators data contains NaN values")
        print(data[data.isnull().any(axis=1)])
        
    return data
