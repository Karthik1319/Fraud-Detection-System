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
    
    # 3. MERCHANT CATEGORIES
    merchant_categories = [
        'grocery', 'electronics', 'gas', 'restaurant', 
        'retail', 'jewelry', 'luxury_goods'
    ]
    merchant_category = np.random.choice(merchant_categories, size=num_rows)
    
    # 4. LOCATION DATA - Customer Homes
    unique_customers = sorted(list(set(customer_ids)))
    customer_home_locations = {}
    
    for cust_id in unique_customers:
        city_choice = np.random.choice(['delhi', 'mumbai', 'bangalore', 'chennai', 'kolkata'])
        if city_choice == 'delhi':
            base_lat, base_long = 28.7041, 77.1025
        elif city_choice == 'mumbai':
            base_lat, base_long = 19.0760, 72.8777
        elif city_choice == 'bangalore':
            base_lat, base_long = 12.9716, 77.5946
        elif city_choice == 'chennai':
            base_lat, base_long = 13.0827, 80.2707
        else:  # kolkata
            base_lat, base_long = 22.5726, 88.3639
        
        # Adding variation (Customers live around the city)
        customer_home_locations[cust_id] = (
            base_lat + np.random.uniform(-0.5, 0.5),
            base_long + np.random.uniform(-0.5, 0.5)
        )

    # Generating Merchant Locations
    merchant_lats = []
    merchant_longs = []
    
    for cust_id in customer_ids:
        home_lat, home_long = customer_home_locations[cust_id]
        
        # 85% transactions near home, 15% travel/fraud
        if np.random.random() < 0.85:
            lat_offset = np.random.normal(0, 0.1)  # around 11km
            long_offset = np.random.normal(0, 0.1)
        else:
            lat_offset = np.random.normal(0, 1.5)  # around 166km
            long_offset = np.random.normal(0, 1.5)
        
        merchant_lat = home_lat + lat_offset
        merchant_long = home_long + long_offset
        
        # Clipping to India bounds
        merchant_lats.append(round(np.clip(merchant_lat, 8.4, 35.5), 4))
        merchant_longs.append(round(np.clip(merchant_long, 68.7, 97.4), 4))

        print("Merchant Location assigned.")
        return pd.DataFrame({'transaction_id': transaction_ids})

if __name__ == "__main__":
    generate_synthetic_data(100)