import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

def generate_synthetic_data(num_rows=100000):
    print(f"Generating {num_rows} synthetic transactions...")
    
    # 1. BASIC IDENTIFIERS
    transaction_ids = [f"TXN_{i:08d}" for i in range(num_rows)]
    customer_ids = [f"CUST_{np.random.randint(1, 5001):05d}" for _ in range(num_rows)]
    card_numbers = [f"CARD_{np.random.randint(10000, 99999)}" for _ in range(num_rows)]
    merchant_ids = [f"MERCHANT_{np.random.randint(1, 1001)}" for _ in range(num_rows)]

    # 2. TIMESTAMPS
    # Generating 90 days of history ending at Sep 15, 2025
    base_date = datetime(2025, 9, 15)
    timestamps = []
    for _ in range(num_rows):
        days_offset = np.random.randint(0, 90)
        hours = np.random.randint(0, 24)
        minutes = np.random.randint(0, 60)
        seconds = np.random.randint(0, 60)
        ts = base_date - timedelta(days=days_offset, hours=hours, minutes=minutes, seconds=seconds)
        timestamps.append(ts)
    
    print("Timestamps generated.")
    return pd.DataFrame({'transaction_id': transaction_ids})

if __name__ == "__main__":
    generate_synthetic_data(100)