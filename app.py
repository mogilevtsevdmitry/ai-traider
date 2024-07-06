import argparse
import numpy as np
import pandas as pd
from data_loader import load_data
from model_trainer import train_model
from predictor import predict
import datetime
from keras.models import load_model

def predict_price(currency_pair):
    # Загрузка последних данных
    end_time = datetime.datetime.now().isoformat()
    data = load_data(currency_pair, '2020-01-01', end_time)
    
    # Загрузка модели и скалера
    scaler = np.load(f'models/{currency_pair}_scaler.npy', allow_pickle=True).item()
    model = load_model(f'models/{currency_pair}_lstm_model.h5')
    
    # Прогнозирование
    predictions = predict(currency_pair, data, scaler)
    
    # Последняя запись данных
    current_time = data['timestamp'].iloc[-1]
    actual_price = data[currency_pair].iloc[-1]
    predicted_price = predictions[-1]
    
    # Определение движения цены
    direction = 'вверх' if predicted_price > actual_price else 'вниз'
    
    # Вычисление вероятности как разницы в ценах, нормированной на диапазон цен за последний период
    max_price = max(data[currency_pair])
    min_price = min(data[currency_pair])
    price_range = max_price - min_price
    if price_range > 0:
        probability = (abs(predicted_price - actual_price) / price_range) * 100
        probability = 100 - probability
    else:
        probability = 100  # Если ценовой диапазон нулевой, вероятность 100%

    # Формирование результата
    result = f"""
    Валютная пара {currency_pair}
    Движение цены {direction}
    Предполагаемая цена через 5 мин {predicted_price:.5f}
    Вероятность {probability:.2f}%
    """
    
    print(result)

def main():
    parser = argparse.ArgumentParser(description='Train or predict currency prices.')
    parser.add_argument('--currency_pair', type=str, required=True, help='Currency pair to analyze')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--start_time', type=str, help='Start time for training data extraction')
    parser.add_argument('--end_time', type=str, help='End time for training data extraction')
    
    args = parser.parse_args()
    
    if args.train:
        if not args.start_time or not args.end_time:
            raise ValueError('Training mode requires --start_time and --end_time')
        data = load_data(args.currency_pair, args.start_time, args.end_time)
        train_model(data, args.currency_pair)
    else:
        predict_price(args.currency_pair)

if __name__ == "__main__":
    main()
