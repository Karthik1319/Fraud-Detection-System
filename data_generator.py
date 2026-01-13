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
    distances_from_home = []
    
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

        # Calculating Distance (Simplified Haversine)
        # 1 deg lat approximates to 111km
        lat_diff = merchant_lat - home_lat
        long_diff = merchant_long - home_long
        
        distance = np.sqrt(
            (lat_diff * 111)**2 + 
            (long_diff * 111 * np.cos(np.radians(home_lat)))**2
        )
        distances_from_home.append(round(max(0, distance), 2))

    # 5. TRANSACTION AMOUNTS
    # Gamma distribution for realistic spread (mean = 2000)
    amounts = np.random.gamma(2, 1000, size=num_rows)
    amounts = np.round(amounts, 2)
    
    # 6. TEMPORAL FEATURES
    hours = [ts.hour for ts in timestamps]
    days_of_week = [ts.weekday() for ts in timestamps]
    months = [ts.month for ts in timestamps]

    # 7. FRAUD INJECTION (Target: 2%)
    is_fraud = np.zeros(num_rows, dtype=int)
    fraud_types = ['none'] * num_rows
    fraud_candidates = set()
    
    hvp = max(1, int(0.006 * num_rows))
    ldp = max(1, int(0.005 * num_rows))
    lnp = max(1, int(0.004 * num_rows))
    hrp = max(1, int(0.005 * num_rows))

    # Pattern 1: High Value
    high_value_mask = amounts > 20000
    fraud_candidates.update(np.where(high_value_mask)[0][:hvp])
    
    # Pattern 2: Large Distance
    large_distance_mask = np.array(distances_from_home) > 100
    fraud_candidates.update(np.where(large_distance_mask)[0][:ldp])
    
    # Pattern 3: Late Night (11PM - 4AM)
    late_night_mask = (np.array(hours) >= 23) | (np.array(hours) <= 4)
    fraud_candidates.update(np.where(late_night_mask)[0][:lnp])

    # Pattern 4: High Risk Categories
    high_risk_mask = np.isin(merchant_category, ['jewelry', 'luxury_goods'])
    fraud_candidates.update(np.where(high_risk_mask)[0][:hrp])

    target_frauds = max(1, int(0.02 * num_rows))
    fraud_candidates = list(fraud_candidates)

    if len(fraud_candidates) < target_frauds:
        remaining = target_frauds - len(fraud_candidates)
        available = set(range(num_rows)) - set(fraud_candidates)
        fraud_candidates.extend(random.sample(list(available), remaining))
    else:
        fraud_candidates = random.sample(fraud_candidates, target_frauds)

    fraud_type_options = ['card_cloning', 'account_takeover', 'merchant_collusion']

    for idx in fraud_candidates:
        is_fraud[idx] = 1

        if amounts[idx] > 30000:
            fraud_types[idx] = 'account_takeover'
            amounts[idx] *= np.random.uniform(2.0, 4.0)
        elif distances_from_home[idx] > 150:
            fraud_types[idx] = 'card_cloning'
            amounts[idx] *= np.random.uniform(1.5, 3.0)
        elif merchant_category[idx] in ['jewelry', 'luxury_goods']:
            fraud_types[idx] = 'merchant_collusion'
        else:
            fraud_types[idx] = random.choice(fraud_type_options)

    # 8. ASSEMBLE DATAFRAME
    df = pd.DataFrame({
        'transaction_id': transaction_ids,
        'customer_id': customer_ids,
        'card_number': card_numbers,
        'timestamp': [ts.strftime('%Y-%m-%dT%H:%M:%SZ') for ts in timestamps],
        'amount': np.round(amounts, 2),
        'merchant_id': merchant_ids,
        'merchant_category': merchant_category,
        'merchant_lat': merchant_lats,
        'merchant_long': merchant_longs,
        'is_fraud': is_fraud,
        'fraud_type': fraud_types,
        'hour': hours,
        'day_of_week': days_of_week,
        'month': months,
        'distance_from_home': distances_from_home
    })

    output_path = 'transactions.csv'
    df.to_csv(output_path, index=False)
    
    print(f"\nSaved to {output_path}")
    print(f"Fraud Rate: {df['is_fraud'].mean():.2%}")
    print(f"Total Frauds: {df['is_fraud'].sum():,}")
    print(df['fraud_type'].value_counts())
    
    return df

if __name__ == "__main__":
    df = generate_synthetic_data(num_rows=100000)
    
    required_columns = [
        'transaction_id', 'customer_id', 'card_number', 'timestamp',
        'amount', 'merchant_id', 'merchant_category', 'merchant_lat',
        'merchant_long', 'is_fraud', 'fraud_type', 'hour', 
        'day_of_week', 'month', 'distance_from_home'
    ]
    
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        print(f"WARNING: Missing columns: {missing_columns}")
    else:
        print("All required columns present!")