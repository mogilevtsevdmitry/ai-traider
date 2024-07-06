import os
import datetime
import numpy as np
import pandas as pd
from keras.models import load_model
from data_loader import load_data
from data_preprocessing import add_indicators
from predictor import predict, calculate_probability
import requests
import logging

# Настройка логирования
logging.basicConfig(filename='analyze_and_notify.log', level=logging.INFO)

# Telegram Bot API settings
TELEGRAM_BOT_TOKEN = '7299237641:AAEfCmy3pqR7SmcIT6lVept7lwHnb5lVjUk'
TELEGRAM_CHAT_ID = '400678398'

currency_pairs = [
    "aud_cad", "eur_usd", "aud_jpy", "nzd_usd", "aud_usd", "usd_jpy", "usd_cad", "usd_chf", "eur_gbp", 
    "eur_jpy", "eur_cad", "gbp_jpy", "eur_aud", "aud_chf", "eur_chf", "chf_jpy", "eur_nzd", "gbp_aud", 
    "cad_chf", "usd_pln", "eur_pln", "usd_mxn", "eur_mxn", "gbp_chf", "aud_nzd", "cad_jpy"
]


def log_message(message):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logging.info(f'{timestamp} - {message}')

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        'chat_id': TELEGRAM_CHAT_ID,
        'text': message,
        'parse_mode': 'HTML'
    }
    response = requests.post(url, data=payload)
    return response.json()

def analyze_currency_pair(currency_pair):
    log_message('Script started')
    end_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    start_time = '2024-06-25' # (datetime.datetime.now() - datetime.timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S')
    
    data = load_data(currency_pair, start_time, end_time)
    if data is None or len(data) == 0:
        print(f"No data for {currency_pair}")
        return False
    
    model_path = os.path.join(os.path.dirname(__file__), 'models', f'{currency_pair}_lstm_model.keras')
    scaler_path = os.path.join(os.path.dirname(__file__), 'models', f'{currency_pair}_scaler.npy')
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print(f"Model or scaler for {currency_pair} not found")
        return False

    model = load_model(model_path)
    scaler = np.load(scaler_path, allow_pickle=True).item()

    # Добавим предобработку данных для совпадения с обучением модели
    data = add_indicators(data, currency_pair)
    data_slice = data.iloc[-60:]
    
    try:
        predicted_prices = predict(currency_pair, data_slice, scaler, model)
        predicted_price = predicted_prices[-1]  # Предполагаем, что результатом является одномерный массив
        actual_price = data[currency_pair].iloc[-1]
        
        # Рассчитываем вероятность
        probability = calculate_probability(predicted_price, actual_price, data_slice, model, scaler, currency_pair)

        direction = 'up' if predicted_price > actual_price else 'down'

        if probability >= 95:
            message = (f"Валютная пара {currency_pair}\n"
                       f"Направление движения {direction}\n"
                       f"Текущая цена {actual_price}\n"
                       f"Вероятность успеха {probability:.2f}%\n"
                       f"Предполагаемая цена {predicted_price}")
            send_telegram_message(message)
            log_message(message)
            return True
    except ValueError as e:
        print(f"Error analyzing {currency_pair}: {e}")

    return False

def main():
    for pair in currency_pairs:
        print(f"Analyzing {pair}...")
        if analyze_currency_pair(pair):
            break  # Если найдена пара с вероятностью больше 95%, выходим из цикла

if __name__ == "__main__":
    main()