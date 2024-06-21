import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt

# Загрузка данных
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')
scaler = np.load('scaler.npy', allow_pickle=True).item()

# Преобразование данных для LSTM
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# Загрузка модели
model = load_model('lstm_model.h5')

# Прогнозирование
predictions = model.predict(X_test)

# Масштабирование предсказанных значений обратно к исходному масштабу
# Создаем массив, который будет иметь такую же форму, как и входные данные для scaler
predicted_prices_scaled = np.zeros((predictions.shape[0], X_test.shape[2]))
predicted_prices_scaled[:, 0] = predictions[:, 0]  # Помещаем предсказания в первый столбец

# Масштабируем обратно только первый столбец
predicted_prices = scaler.inverse_transform(predicted_prices_scaled)[:, 0]
actual_prices = scaler.inverse_transform(np.column_stack((y_test, np.zeros((y_test.shape[0], X_test.shape[2] - 1)))))[:, 0]

# Построение графика
plt.figure(figsize=(14, 5))
plt.plot(actual_prices, color='blue', label='Actual Prices')
plt.plot(predicted_prices, color='red', label='Predicted Prices')
plt.title('EUR/USD Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
