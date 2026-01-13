import requests
import json
from datetime import datetime

API_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("=" * 80)
    print("Testing Health Endpoint...")
    print("=" * 80)
    
    response = requests.get(f"{API_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_legitimate_transaction():
    """Test with a legitimate transaction"""
    print("=" * 80)
    print("Testing Legitimate Transaction...")
    print("=" * 80)
    
    transaction = {
        "transaction_id": "TXN_LEGIT_001",
        "customer_id": "CUST_00001",
        "card_number": "CARD_12345",
        "timestamp": "2025-09-15T14:32:45Z",
        "amount": 2500.00,
        "merchant_id": "MERCHANT_1234",
        "merchant_category": "grocery",
        "merchant_lat": 28.5355,
        "merchant_long": 77.3910,
        "distance_from_home": 5.0
    }
    
    print(f"Input: {json.dumps(transaction, indent=2)}")
    
    response = requests.post(f"{API_URL}/predict", json=transaction)
    print(f"\nStatus Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_fraudulent_transaction():
    """Test with a suspicious/fraudulent transaction"""
    print("=" * 80)
    print("Testing Fraudulent Transaction...")
    print("=" * 80)
    
    transaction = {
        "transaction_id": "TXN_FRAUD_001",
        "customer_id": "CUST_00002",
        "card_number": "CARD_67890",
        "timestamp": "2025-09-16T02:30:00Z",  # Late night
        "amount": 150000.00,  # Very high amount
        "merchant_id": "MERCHANT_5678",
        "merchant_category": "jewelry",  # High-risk
        "merchant_lat": 18.5204,
        "merchant_long": 73.8567,
        "distance_from_home": 350.0  # Very far
    }
    
    print(f"Input: {json.dumps(transaction, indent=2)}")
    
    response = requests.post(f"{API_URL}/predict", json=transaction)
    print(f"\nStatus Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_performance():
    """Test prediction performance (latency)"""
    print("=" * 80)
    print("Testing Performance (10 predictions)...")
    print("=" * 80)
    
    latencies = []
    
    for i in range(10):
        transaction = {
            "transaction_id": f"TXN_PERF_{i:03d}",
            "customer_id": f"CUST_{i:05d}",
            "card_number": f"CARD_{i:05d}",
            "timestamp": "2025-09-15T14:32:45Z",
            "amount": 5000.00 + i * 100,
            "merchant_id": "MERCHANT_1234",
            "merchant_category": "electronics",
            "merchant_lat": 28.5355,
            "merchant_long": 77.3910,
            "distance_from_home": 12.5
        }
        
        response = requests.post(f"{API_URL}/predict", json=transaction)
        result = response.json()
        latencies.append(result['prediction_time_ms'])
        print(f"   Prediction {i+1}: {result['prediction_time_ms']:.2f} ms")
    
    avg_latency = sum(latencies) / len(latencies)
    max_latency = max(latencies)
    
    print(f"\nAverage Latency: {avg_latency:.2f} ms")
    print(f"Max Latency: {max_latency:.2f} ms")
    print(f"Target: <100 ms")
    print(f"Status: {'PASS' if max_latency < 100 else '❌ FAIL'}")
    print()

def main():
    """Run all tests"""
    print("\n")
    print("*" * 80)
    print("*" + " " * 78 + "*")
    print("*" + " " * 20 + "FRAUD DETECTION API TESTS" + " " * 33 + "*")
    print("*" + " " * 78 + "*")
    print("*" * 80)
    print("\n")
    
    try:
        test_health()
        test_legitimate_transaction()
        test_fraudulent_transaction()
        test_performance()
        
        print("=" * 80)
        print("ALL TESTS COMPLETED SUCCESSFULLY! ✅")
        print("=" * 80)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Cannot connect to API")
        print("Make sure the API is running:")
        print("   python app.py")
        print("\nOr start with uvicorn:")
        print("   uvicorn app:app --reload")
        print()
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        print()

if __name__ == "__main__":
    main()