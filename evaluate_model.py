import os
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from data_loader import load_data
from data_preprocessing import add_indicators
from keras.models import load_model
from datetime import datetime

def evaluate_model(currency_pair, start_time, end_time):
    # Загрузка данных
    data = load_data(currency_pair, start_time, end_time)
    if data is None or len(data) == 0:
        print(f"No data for {currency_pair}")
        return
    
    # Загрузка модели и скейлера
    model_path = f'models/{currency_pair}_lstm_model.keras'
    scaler_path = f'models/{currency_pair}_scaler.npy'
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print(f"Model or scaler for {currency_pair} not found")
        return

    model = load_model(model_path)
    scaler = np.load(scaler_path, allow_pickle=True).item()

    # Добавление индикаторов
    data = add_indicators(data, currency_pair)
    # Обработка NaN значений
    data = data.fillna(method='ffill').fillna(method='bfill')
    
    # Подготовка данных
    data_scaled = scaler.transform(data)
    X_test = []
    y_test = []
    for i in range(60, len(data_scaled)):
        X_test.append(data_scaled[i-60:i])
        y_test.append(data_scaled[i, 0])
    
    X_test, y_test = np.array(X_test), np.array(y_test)
    
    # Предсказания модели
    y_pred = model.predict(X_test)
    
    # Вычисление метрик
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"Evaluation for {currency_pair}:")
    print(f"MAE: {mae}")
    print(f"RMSE: {rmse}")
    print(f"R^2: {r2}")
    
    return mae, rmse, r2

if __name__ == "__main__":
    currency_pair = "aud_cad"
    start_time = "2024-06-25"
    end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    evaluate_model(currency_pair, start_time, end_time)
