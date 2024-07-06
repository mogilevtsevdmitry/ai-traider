.PHONY: train predict

train:
	python app.py --currency_pair eur_usd --start_time "2024-06-25" --end_time "2024-07-03" --train

predict:
	python app.py --currency_pair eur_usd
