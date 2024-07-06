# model_trainer.py
import numpy as np
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Dropout, Input
from sklearn.preprocessing import MinMaxScaler
from data_preprocessing import add_indicators
import os

def train_model(data, currency_pair, epochs=50, batch_size=32, units=50, dropout_rate=0.2):
    data = add_indicators(data, currency_pair)
    
    # Проверка и заполнение NaN значений
    if data.isna().sum().sum() > 0:
        print("Indicators data contains NaN values")
        print(data)
        data = data.fillna(method='ffill').fillna(method='bfill')
    
    # Повторная проверка на NaN
    if data.isna().sum().sum() > 0:
        print("NaN values still present after filling")
        data = data.dropna()

    print("Data after handling NaN values:")
    print(data.head())

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    
    X = []
    y = []
    for i in range(60, len(data_scaled)):
        X.append(data_scaled[i-60:i])
        y.append(data_scaled[i, 0])
    
    X, y = np.array(X), np.array(y)
    
    inputs = Input(shape=(X.shape[1], X.shape[2]))
    x = LSTM(units=units, return_sequences=True)(inputs)
    x = Dropout(dropout_rate)(x)
    x = LSTM(units=units)(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(1)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=epochs, batch_size=batch_size)
    
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_path = os.path.join(model_dir, f'{currency_pair}_lstm_model.keras')
    scaler_path = os.path.join(model_dir, f'{currency_pair}_scaler.npy')
    
    model.save(model_path)  # Использование нового формата для сохранения модели
    np.save(scaler_path, scaler)
    
    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")
    
    return model, scaler
