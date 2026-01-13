import pytest
from fastapi.testclient import TestClient
from app import app
import json

client = TestClient(app)


class TestAPIEndpoints:
    """Test API endpoints"""
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"
        assert "endpoints" in data
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "models_loaded" in data
    
    def test_predict_valid_transaction(self):
        """Test prediction with valid transaction"""
        transaction_data = {
            "transaction_id": "TXN_TEST_001",
            "customer_id": "CUST_00001",
            "card_number": "CARD_12345",
            "timestamp": "2025-09-15T14:32:45Z",
            "amount": 5000.00,
            "merchant_id": "MERCHANT_1234",
            "merchant_category": "electronics",
            "merchant_lat": 28.5355,
            "merchant_long": 77.3910,
            "distance_from_home": 12.5
        }
        
        response = client.post("/predict", json=transaction_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "transaction_id" in data
        assert "is_fraud" in data
        assert "fraud_probability" in data
        assert "confidence" in data
        assert "reasoning" in data
        assert "prediction_time_ms" in data
        
        # Check prediction time requirement (<100ms)
        assert data["prediction_time_ms"] < 100, f"Prediction took {data['prediction_time_ms']}ms"
    
    def test_predict_high_risk_transaction(self):
        """Test prediction with high-risk transaction"""
        transaction_data = {
            "transaction_id": "TXN_HIGH_RISK",
            "customer_id": "CUST_00002",
            "card_number": "CARD_67890",
            "timestamp": "2025-09-15T23:45:00Z",  # Late night
            "amount": 150000.00,  # Very high amount
            "merchant_id": "MERCHANT_5678",
            "merchant_category": "jewelry",  # High-risk category
            "merchant_lat": 28.5355,
            "merchant_long": 77.3910,
            "distance_from_home": 300.0  # Very far from home
        }
        
        response = client.post("/predict", json=transaction_data)
        assert response.status_code == 200
        
        data = response.json()
        # High-risk transaction should have higher fraud probability
        assert data["fraud_probability"] > 0.5
        assert len(data["reasoning"]["risk_factors"]) > 0
    
    def test_predict_low_risk_transaction(self):
        """Test prediction with low-risk transaction"""
        transaction_data = {
            "transaction_id": "TXN_LOW_RISK",
            "customer_id": "CUST_00003",
            "card_number": "CARD_11111",
            "timestamp": "2025-09-15T12:00:00Z",  # Business hours
            "amount": 500.00,  # Low amount
            "merchant_id": "MERCHANT_9999",
            "merchant_category": "grocery",  # Low-risk category
            "merchant_lat": 28.5355,
            "merchant_long": 77.3910,
            "distance_from_home": 2.0  # Close to home
        }
        
        response = client.post("/predict", json=transaction_data)
        assert response.status_code == 200
        
        data = response.json()
        # Low-risk transaction should have lower fraud probability
        assert data["fraud_probability"] < 0.7


class TestInputValidation:
    """Test input validation"""
    
    def test_invalid_amount(self):
        """Test with negative amount"""
        transaction_data = {
            "transaction_id": "TXN_INVALID",
            "customer_id": "CUST_00001",
            "card_number": "CARD_12345",
            "timestamp": "2025-09-15T14:32:45Z",
            "amount": -100.00,  # Invalid negative amount
            "merchant_id": "MERCHANT_1234",
            "merchant_category": "electronics",
            "merchant_lat": 28.5355,
            "merchant_long": 77.3910,
            "distance_from_home": 12.5
        }
        
        response = client.post("/predict", json=transaction_data)
        assert response.status_code == 422  # Validation error
    
    def test_invalid_category(self):
        """Test with invalid merchant category"""
        transaction_data = {
            "transaction_id": "TXN_INVALID",
            "customer_id": "CUST_00001",
            "card_number": "CARD_12345",
            "timestamp": "2025-09-15T14:32:45Z",
            "amount": 5000.00,
            "merchant_id": "MERCHANT_1234",
            "merchant_category": "invalid_category",
            "merchant_lat": 28.5355,
            "merchant_long": 77.3910,
            "distance_from_home": 12.5
        }
        
        response = client.post("/predict", json=transaction_data)
        assert response.status_code == 422
    
    def test_invalid_coordinates(self):
        """Test with invalid coordinates"""
        transaction_data = {
            "transaction_id": "TXN_INVALID",
            "customer_id": "CUST_00001",
            "card_number": "CARD_12345",
            "timestamp": "2025-09-15T14:32:45Z",
            "amount": 5000.00,
            "merchant_id": "MERCHANT_1234",
            "merchant_category": "electronics",
            "merchant_lat": 100.0,  # Invalid latitude
            "merchant_long": 77.3910,
            "distance_from_home": 12.5
        }
        
        response = client.post("/predict", json=transaction_data)
        assert response.status_code == 422
    
    def test_missing_required_field(self):
        """Test with missing required field"""
        transaction_data = {
            "transaction_id": "TXN_INVALID",
            "customer_id": "CUST_00001",
            # Missing card_number
            "timestamp": "2025-09-15T14:32:45Z",
            "amount": 5000.00,
            "merchant_id": "MERCHANT_1234",
            "merchant_category": "electronics",
            "merchant_lat": 28.5355,
            "merchant_long": 77.3910,
            "distance_from_home": 12.5
        }
        
        response = client.post("/predict", json=transaction_data)
        assert response.status_code == 422


class TestEdgeCases:
    """Test edge cases"""
    
    def test_very_high_amount(self):
        """Test with extremely high amount"""
        transaction_data = {
            "transaction_id": "TXN_EXTREME",
            "customer_id": "CUST_00001",
            "card_number": "CARD_12345",
            "timestamp": "2025-09-15T14:32:45Z",
            "amount": 500000.00,  # 5 lakh INR
            "merchant_id": "MERCHANT_1234",
            "merchant_category": "luxury_goods",
            "merchant_lat": 28.5355,
            "merchant_long": 77.3910,
            "distance_from_home": 12.5
        }
        
        response = client.post("/predict", json=transaction_data)
        assert response.status_code == 200
        assert response.json()["fraud_probability"] > 0.5
    
    def test_zero_distance(self):
        """Test with zero distance from home"""
        transaction_data = {
            "transaction_id": "TXN_HOME",
            "customer_id": "CUST_00001",
            "card_number": "CARD_12345",
            "timestamp": "2025-09-15T14:32:45Z",
            "amount": 1000.00,
            "merchant_id": "MERCHANT_1234",
            "merchant_category": "grocery",
            "merchant_lat": 28.5355,
            "merchant_long": 77.3910,
            "distance_from_home": 0.0
        }
        
        response = client.post("/predict", json=transaction_data)
        assert response.status_code == 200
    
    def test_late_night_transaction(self):
        """Test late night transaction"""
        transaction_data = {
            "transaction_id": "TXN_LATE_NIGHT",
            "customer_id": "CUST_00001",
            "card_number": "CARD_12345",
            "timestamp": "2025-09-16T02:30:00Z",  # 2:30 AM
            "amount": 80000.00,
            "merchant_id": "MERCHANT_1234",
            "merchant_category": "electronics",
            "merchant_lat": 28.5355,
            "merchant_long": 77.3910,
            "distance_from_home": 200.0
        }
        
        response = client.post("/predict", json=transaction_data)
        assert response.status_code == 200
        data = response.json()
        # Should have "Late night transaction" in risk factors
        risk_factors = data["reasoning"]["risk_factors"]
        assert any("Late night" in factor for factor in risk_factors)


class TestPerformance:
    """Test performance requirements"""
    
    def test_prediction_latency(self):
        """Test prediction meets latency requirement (<100ms)"""
        transaction_data = {
            "transaction_id": "TXN_PERF",
            "customer_id": "CUST_00001",
            "card_number": "CARD_12345",
            "timestamp": "2025-09-15T14:32:45Z",
            "amount": 5000.00,
            "merchant_id": "MERCHANT_1234",
            "merchant_category": "electronics",
            "merchant_lat": 28.5355,
            "merchant_long": 77.3910,
            "distance_from_home": 12.5
        }
        
        response = client.post("/predict", json=transaction_data)
        assert response.status_code == 200
        
        prediction_time = response.json()["prediction_time_ms"]
        assert prediction_time < 100, f"Prediction took {prediction_time}ms, exceeds 100ms requirement"
    
    def test_batch_predictions(self):
        """Test multiple predictions in sequence"""
        transaction_base = {
            "customer_id": "CUST_00001",
            "card_number": "CARD_12345",
            "timestamp": "2025-09-15T14:32:45Z",
            "amount": 5000.00,
            "merchant_id": "MERCHANT_1234",
            "merchant_category": "electronics",
            "merchant_lat": 28.5355,
            "merchant_long": 77.3910,
            "distance_from_home": 12.5
        }
        
        # Send 10 predictions
        latencies = []
        for i in range(10):
            transaction_data = transaction_base.copy()
            transaction_data["transaction_id"] = f"TXN_BATCH_{i}"
            
            response = client.post("/predict", json=transaction_data)
            assert response.status_code == 200
            latencies.append(response.json()["prediction_time_ms"])
        
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        
        assert avg_latency < 100, f"Average latency {avg_latency:.2f}ms exceeds 100ms"
        assert max_latency < 100, f"Max latency {max_latency:.2f}ms exceeds 100ms"
        
        print(f"\nPerformance: Avg={avg_latency:.2f}ms, Max={max_latency:.2f}ms")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])