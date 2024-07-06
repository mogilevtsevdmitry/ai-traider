import os
import datetime
from data_loader import load_data
from model_trainer import train_model
from analyze_and_notify import analyze_currency_pair

currency_pairs = [
    "aud_cad", "eur_usd", "aud_jpy", "nzd_usd", "aud_usd", "usd_jpy", "usd_cad", "usd_chf", "eur_gbp", 
    "eur_jpy", "eur_cad", "gbp_jpy", "eur_aud", "aud_chf", "eur_chf", "chf_jpy", "eur_nzd", "gbp_aud", 
    "cad_chf", "usd_pln", "eur_pln", "usd_mxn", "eur_mxn", "gbp_chf", "aud_nzd", "cad_jpy"
]

def check_data_for_nan(data):
    if data.isnull().values.any():
        print("Data contains NaN values")
        print(data[data.isnull().any(axis=1)])

def main():
    end_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    start_time = '2024-06-25'
    
    for pair in currency_pairs:
        print(f"Training model for {pair}...")
        data = load_data(pair, start_time, end_time)
        
        if data is None or data.empty:
            print(f"No data for {pair}")
            continue
        check_data_for_nan(data)
        
        if data is not None:
            train_model(data, pair)
        else:
            print(f"Skipping {pair} due to missing data")
    
    os.system('python -m unittest test_predictor.py') 

if __name__ == "__main__":
    main()
