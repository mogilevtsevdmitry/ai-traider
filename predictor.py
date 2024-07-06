import numpy as np

def predict(currency_pair, data, scaler, model):
    features = [
        'SMA', 'EMA', 'RSI', 'MACD', 'MACD_signal', 'MACD_diff', 
        'Bollinger_high', 'Bollinger_low', 'Stochastic', 'ATR', 
        'ADX', 'CCI', 'ROC', 'WilliamsR', currency_pair
    ]
    
    X = data[features].values
    X_scaled = scaler.transform(X)
    
    # Модель ожидает входы формы (samples, timesteps, features)
    # Разделим данные на отрезки длиной 60 для входа в LSTM
    X_segments = []
    segment_length = 60
    for i in range(len(X_scaled) - segment_length + 1):
        X_segments.append(X_scaled[i: i + segment_length])

    X_segments = np.array(X_segments)
    
    predicted_prices = model.predict(X_segments)
    return predicted_prices

def calculate_probability(predicted_price, actual_price, test_data, model, scaler, currency_pair):
    test_data = test_data[-60:]  # Берем последние 60 записей для входа в модель
    features = [
        'SMA', 'EMA', 'RSI', 'MACD', 'MACD_signal', 'MACD_diff', 
        'Bollinger_high', 'Bollinger_low', 'Stochastic', 'ATR', 
        'ADX', 'CCI', 'ROC', 'WilliamsR', currency_pair
    ]
    X_test = test_data[features].values
    X_test_scaled = scaler.transform(X_test)
    X_test_segments = np.array([X_test_scaled])  # Превращаем в трехмерный массив

    test_predictions = model.predict(X_test_segments)
    prediction_diff = np.abs(test_predictions - actual_price)
    probability = max(0, 100 - (np.mean(prediction_diff) * 100))

    return probability
