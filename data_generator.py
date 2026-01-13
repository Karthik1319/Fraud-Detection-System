import pandas as pd
import numpy as np
import random

# Random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_synthetic_data(num_rows=100000):
    print(f"Generating {num_rows} synthetic transactions...")
    
    # BASIC IDENTIFIERS
    # Generating unique IDs for transactions, customers, and cards
    transaction_ids = [f"TXN_{i:08d}" for i in range(num_rows)]
    customer_ids = [f"CUST_{np.random.randint(1, 5001):05d}" for _ in range(num_rows)]  # 5000 unique
    card_numbers = [f"CARD_{np.random.randint(10000, 99999)}" for _ in range(num_rows)]
    merchant_ids = [f"MERCHANT_{np.random.randint(1, 1001)}" for _ in range(num_rows)]  # 1000 merchants

    print("IDs generated.")
    return pd.DataFrame({'transaction_id': transaction_ids})

if __name__ == "__main__":
    generate_synthetic_data(100)