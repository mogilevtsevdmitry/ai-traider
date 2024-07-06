import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.exc import ProgrammingError

db_config = {
    'user': 'traider',
    'password': '123456traider',
    'host': 'localhost',
    'port': '5435',
    'database': 'db_traider'
}

def load_data(currency_pair, start_time, end_time=None):
    engine = create_engine(f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}")
    query = f"""
    SELECT timestamp, {currency_pair}
    FROM couples_relationship
    WHERE timestamp >= '{start_time}'
    """
    if end_time:
        query += f" AND timestamp <= '{end_time}'"
    query += " ORDER BY timestamp"
    
    try:
        data = pd.read_sql(query, engine)
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data.set_index('timestamp', inplace=True)
        return data
    except ProgrammingError as e:
        print(f"Error: {e}")
        return None
