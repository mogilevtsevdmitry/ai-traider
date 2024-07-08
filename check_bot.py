import requests

# Вставьте ваш токен API бота
TOKEN = ''

# Получение обновлений от бота
url = f"https://api.telegram.org/bot{TOKEN}/getUpdates"
response = requests.get(url)
data = response.json()

# Найдите идентификатор чата в ответе
for result in data['result']:
    print(result['message']['chat']['id'])
