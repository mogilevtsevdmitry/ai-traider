import requests

# Вставьте ваш токен API бота
TOKEN = '7299237641:AAEfCmy3pqR7SmcIT6lVept7lwHnb5lVjUk'

# Получение обновлений от бота
url = f"https://api.telegram.org/bot{TOKEN}/getUpdates"
response = requests.get(url)
data = response.json()

# Найдите идентификатор чата в ответе
for result in data['result']:
    print(result['message']['chat']['id'])
