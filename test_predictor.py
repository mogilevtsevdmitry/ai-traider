import os
import unittest
import numpy as np
import pandas as pd
import json
import requests
from datetime import datetime
from data_loader import load_data
from data_preprocessing import add_indicators
from predictor import predict
from keras.models import load_model

currency_pairs = [
    "aud_cad", "eur_usd", "aud_jpy"#, "nzd_usd", "aud_usd", "usd_jpy", "usd_cad", "usd_chf", "eur_gbp"
    # "eur_jpy", "eur_cad", "gbp_jpy", "eur_aud", "aud_chf", "eur_chf", "chf_jpy", "eur_nzd", "gbp_aud", 
    # "cad_chf", "usd_pln", "eur_pln", "usd_mxn", "eur_mxn", "gbp_chf", "aud_nzd", "cad_jpy"
]

TELEGRAM_BOT_TOKEN = '7299237641:AAEfCmy3pqR7SmcIT6lVept7lwHnb5lVjUk'
TELEGRAM_CHAT_ID = '400678398'

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        'chat_id': TELEGRAM_CHAT_ID,
        'text': message,
        'parse_mode': 'HTML'
    }
    response = requests.post(url, data=payload)
    return response.json()

class TestPredictor(unittest.TestCase):
    def test_predictions(self):
        overall_results = []

        end_date = datetime.now().strftime('%Y-%m-%d')

        for currency_pair in currency_pairs:
            print(f'Testing {currency_pair}...')
            data = load_data(currency_pair, '2024-06-25', end_date)  # Загрузите все данные за период
            
            if data is None or data.empty:
                print(f"No data for {currency_pair}")
                continue
            
            model_path = f'models/{currency_pair}_lstm_model.keras'
            scaler_path = f'models/{currency_pair}_scaler.npy'
            
            if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                print(f"Model or scaler for {currency_pair} not found")
                continue
            
            model = load_model(model_path)
            scaler = np.load(scaler_path, allow_pickle=True).item()

            # Добавим предобработку данных для совпадения с обучением модели
            data = add_indicators(data, currency_pair)
            data = data.fillna(method='ffill').fillna(method='bfill')
            data = data.dropna()

            start_index = data.index[0] + pd.Timedelta(days=1)  # Начальная дата + 1 сутки
            interval = pd.Timedelta(minutes=10)
            prediction_interval = pd.Timedelta(minutes=5)

            correct_predictions = 0
            total_predictions = 0
            detailed_results = []

            while start_index + prediction_interval < data.index[-1]:
                try:
                    data_slice = data.loc[:start_index]  # Данные до текущей точки
                    actual_price_5min = data.loc[start_index + prediction_interval, currency_pair].item()
                    last_actual_price = data_slice[currency_pair].iloc[-1].item()
                except (KeyError, ValueError):
                    # Пропуск итерации, если нет данных для текущей метки времени
                    start_index += interval
                    continue

                predicted_prices = predict(currency_pair, data_slice, scaler, model)
                predicted_price = predicted_prices[-1][0]  # Получение последнего предсказания
                
                direction_predicted = 'up' if predicted_price > last_actual_price else 'down'
                direction_actual = 'up' if actual_price_5min > last_actual_price else 'down'

                if direction_predicted == direction_actual:
                    detailed_results.append({
                        "timestamp": start_index,
                        "predicted_price": predicted_price,
                        "actual_price_5min": actual_price_5min,
                        "direction_predicted": direction_predicted,
                        "direction_actual": direction_actual,
                        "correct": direction_predicted == direction_actual
                    })
                    correct_predictions += 1

                total_predictions += 1
                start_index += interval

            success_rate = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
            if correct_predictions == total_predictions and total_predictions > 0:
                overall_results.append({
                    "currency_pair": currency_pair,
                    "total_predictions": total_predictions,
                    "success_rate": success_rate,
                    "detailed_results": detailed_results
                })

            print(f'Валютная пара {currency_pair}: Проведено предположений: {total_predictions}, Успешных предположений: {success_rate:.2f}%')

        # Сохранение результатов в файл
        with open('prediction_results.json', 'w') as f:
            json.dump(overall_results, f, indent=4, default=str)

        message = "Результаты тестирования моделей:\n\n"
        total_success_rate = 0
        pairs_with_predictions = 0

        for result in overall_results:
            if result['total_predictions'] > 0:
                total_success_rate += result['success_rate']
                pairs_with_predictions += 1
            message += (f"Валютная пара {result['currency_pair']}:\n"
                        f"Проведено предположений: {result['total_predictions']}\n"
                        f"Успешных предположений: {result['success_rate']:.2f}%\n\n")

        if pairs_with_predictions > 0:
            overall_success_rate = total_success_rate / pairs_with_predictions
            message += f"Общий итог: {overall_success_rate:.2f}% успешных предположений\n"

        send_telegram_message(message)

if __name__ == '__main__':
    unittest.main()