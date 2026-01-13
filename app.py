from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Dict, Any
import numpy as np
import pandas as pd
import joblib
import logging
from datetime import datetime
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time anomaly detection for transaction fraud",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global variables for models
MODELS = {}
SCALER = None
FEATURE_COLUMNS = None

# Model paths
MODEL_DIR = Path("models")


class TransactionInput(BaseModel):
    """Input schema for transaction prediction"""
    transaction_id: str = Field(..., description="Unique transaction identifier")
    customer_id: str = Field(..., description="Customer identifier")
    card_number: str = Field(..., description="Card number (hashed)")
    timestamp: str = Field(..., description="Transaction timestamp (ISO 8601)")
    amount: float = Field(..., gt=0, description="Transaction amount in INR")
    merchant_id: str = Field(..., description="Merchant identifier")
    merchant_category: str = Field(..., description="Merchant category")
    merchant_lat: float = Field(..., ge=-90, le=90, description="Merchant latitude")
    merchant_long: float = Field(..., ge=-180, le=180, description="Merchant longitude")
    distance_from_home: float = Field(..., ge=0, description="Distance from customer home (km)")
    
    @validator('merchant_category')
    def validate_category(cls, v):
        valid_categories = [
            'grocery', 'electronics', 'gas', 'restaurant', 
            'retail', 'jewelry', 'luxury_goods'
        ]
        if v not in valid_categories:
            raise ValueError(f"Invalid category. Must be one of: {valid_categories}")
        return v
    
    @validator('timestamp')
    def validate_timestamp(cls, v):
        try:
            datetime.fromisoformat(v.replace('Z', '+00:00'))
        except:
            raise ValueError("Invalid timestamp format. Use ISO 8601 format.")
        return v

    class Config:
        schema_extra = {
            "example": {
                "transaction_id": "TXN_00000001",
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
        }


class PredictionResponse(BaseModel):
    """Response schema for fraud prediction"""
    transaction_id: str
    is_fraud: bool
    fraud_probability: float
    confidence: str
    reasoning: Dict[str, Any]
    models_agreement: Dict[str, bool]
    prediction_time_ms: float
    timestamp: str


def load_models():
    """Load all trained models and preprocessing objects"""
    global MODELS, SCALER, FEATURE_COLUMNS
    
    try:
        logger.info("Loading models...")
        
        # Load Isolation Forest
        MODELS['isolation_forest'] = joblib.load(MODEL_DIR / 'isolation_forest.pkl')
        logger.info("✓ Isolation Forest loaded")
        
        # Load One-Class SVM
        MODELS['one_class_svm'] = joblib.load(MODEL_DIR / 'one_class_svm.pkl')
        logger.info("✓ One-Class SVM loaded")
        
        # Load Autoencoder
        try:
            import tensorflow as tf
            MODELS['autoencoder'] = tf.keras.models.load_model(MODEL_DIR / 'autoencoder.h5')
            MODELS['ae_threshold'] = joblib.load(MODEL_DIR / 'autoencoder_threshold.pkl')['threshold']
            logger.info("✓ Autoencoder loaded")
        except Exception as e:
            logger.warning(f"Autoencoder not loaded: {e}")
        
        # Load preprocessing objects
        SCALER = joblib.load(MODEL_DIR / 'scaler.pkl')
        FEATURE_COLUMNS = joblib.load(MODEL_DIR / 'feature_columns.pkl')
        
        logger.info("✓ All models and preprocessing objects loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise


def engineer_features(transaction: TransactionInput) -> pd.DataFrame:
    """Apply feature engineering to transaction data"""
    try:
        # Parse timestamp
        dt = datetime.fromisoformat(transaction.timestamp.replace('Z', '+00:00'))
        
        # Create base features
        data = {
            'amount': transaction.amount,
            'merchant_lat': transaction.merchant_lat,
            'merchant_long': transaction.merchant_long,
            'distance_from_home': transaction.distance_from_home,
            'hour': dt.hour,
            'day_of_week': dt.weekday(),
            'month': dt.month,
        }
        
        df = pd.DataFrame([data])
        
        # Feature engineering (must match training pipeline exactly!)
        df['amount_log'] = np.log1p(df['amount'])
        df['amount_squared'] = df['amount'] ** 2
        
        # Categorical features
        category_mapping = {
            'grocery': 0, 'electronics': 1, 'gas': 2, 'restaurant': 3,
            'retail': 4, 'jewelry': 5, 'luxury_goods': 6
        }
        df['merchant_category_encoded'] = category_mapping[transaction.merchant_category]
        df['is_high_risk_category'] = int(transaction.merchant_category in ['jewelry', 'luxury_goods'])
        
        # Temporal features
        df['is_late_night'] = int((dt.hour >= 23) or (dt.hour <= 4))
        df['is_weekend'] = int(dt.weekday() >= 5)
        df['is_business_hours'] = int((dt.hour >= 9) and (dt.hour <= 17))
        
        # Distance features
        df['distance_risk'] = int(df['distance_from_home'].iloc[0] > 50)
        df['distance_log'] = np.log1p(df['distance_from_home'])
        
        # Interaction features
        df['amount_distance_interaction'] = df['amount'] * df['distance_from_home']
        df['high_amount_late_night'] = int(
            (df['amount'].iloc[0] > 20000) and (df['is_late_night'].iloc[0] == 1)
        )
        
        # Ensure correct column order
        df = df[FEATURE_COLUMNS]
        
        return df
        
    except Exception as e:
        logger.error(f"Error in feature engineering: {str(e)}")
        raise


def get_fraud_reasoning(transaction: TransactionInput, features: pd.DataFrame, 
                        predictions: Dict[str, bool], scores: Dict[str, float]) -> Dict[str, Any]:
    """Generate human-readable reasoning for fraud prediction"""
    
    risk_factors = []
    
    # High amount
    if transaction.amount > 20000:
        risk_factors.append(f"High transaction amount: ₹{transaction.amount:,.2f}")
    
    # High-risk category
    if transaction.merchant_category in ['jewelry', 'luxury_goods']:
        risk_factors.append(f"High-risk merchant category: {transaction.merchant_category}")
    
    # Late night transaction
    dt = datetime.fromisoformat(transaction.timestamp.replace('Z', '+00:00'))
    if dt.hour >= 23 or dt.hour <= 4:
        risk_factors.append(f"Late night transaction: {dt.hour}:00 hours")
    
    # Long distance from home
    if transaction.distance_from_home > 50:
        risk_factors.append(f"Far from home: {transaction.distance_from_home:.1f} km")
    
    # Unusual location
    if features['distance_risk'].iloc[0] == 1:
        risk_factors.append("Unusual geographic location")
    
    # Model consensus
    fraud_count = sum(predictions.values())
    
    reasoning = {
        "risk_factors": risk_factors if risk_factors else ["No significant risk factors detected"],
        "model_consensus": f"{fraud_count}/{len(predictions)} models flagged as fraud",
        "anomaly_scores": {
            k: round(float(v), 4) for k, v in scores.items()
        },
        "transaction_profile": {
            "amount": transaction.amount,
            "category": transaction.merchant_category,
            "time": dt.strftime("%H:%M"),
            "distance_km": transaction.distance_from_home
        }
    }
    
    return reasoning


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    import os
    os.makedirs('logs', exist_ok=True)
    load_models()
    logger.info("API startup complete")


@app.get("/")
async def root():
    """Root endpoint - API information"""
    return {
        "service": "Fraud Detection API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
            "docs": "/docs"
        },
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    models_loaded = all([
        MODELS.get('isolation_forest') is not None,
        MODELS.get('one_class_svm') is not None,
        SCALER is not None,
        FEATURE_COLUMNS is not None
    ])
    
    return {
        "status": "healthy" if models_loaded else "unhealthy",
        "models_loaded": {
            "isolation_forest": MODELS.get('isolation_forest') is not None,
            "one_class_svm": MODELS.get('one_class_svm') is not None,
            "autoencoder": MODELS.get('autoencoder') is not None,
            "scaler": SCALER is not None
        },
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_fraud(transaction: TransactionInput):
    """
    Predict fraud probability for a transaction
    
    Returns fraud probability, confidence level, and detailed reasoning
    """
    start_time = time.time()
    
    try:
        logger.info(f"Prediction request for transaction: {transaction.transaction_id}")
        
        # Feature engineering
        features = engineer_features(transaction)
        features_scaled = SCALER.transform(features)
        
        # Get predictions from all models
        predictions = {}
        scores = {}
        
        # Isolation Forest
        iso_pred = MODELS['isolation_forest'].predict(features_scaled)[0]
        iso_score = -MODELS['isolation_forest'].score_samples(features_scaled)[0]
        predictions['isolation_forest'] = (iso_pred == -1)
        scores['isolation_forest'] = iso_score
        
        # One-Class SVM
        svm_pred = MODELS['one_class_svm'].predict(features_scaled)[0]
        svm_score = -MODELS['one_class_svm'].decision_function(features_scaled)[0]
        predictions['one_class_svm'] = (svm_pred == -1)
        scores['one_class_svm'] = svm_score
        
        # Autoencoder (if available)
        if MODELS.get('autoencoder'):
            reconstruction = MODELS['autoencoder'].predict(features_scaled, verbose=0)
            ae_score = np.mean(np.abs(features_scaled - reconstruction))
            predictions['autoencoder'] = (ae_score > MODELS['ae_threshold'])
            scores['autoencoder'] = ae_score
        
        # Ensemble decision (majority voting)
        fraud_votes = sum(predictions.values())
        is_fraud = fraud_votes >= 2
        
        # Calculate fraud probability (weighted average of normalized scores)
        iso_prob = min(iso_score / 1.0, 1.0)
        svm_prob = min(svm_score / 2.0, 1.0)
        
        if MODELS.get('autoencoder'):
            ae_prob = min(ae_score / MODELS['ae_threshold'], 1.0)
            fraud_probability = (iso_prob + svm_prob + ae_prob) / 3
        else:
            fraud_probability = (iso_prob + svm_prob) / 2
        
        # Determine confidence
        if fraud_votes == len(predictions):
            confidence = "HIGH"
        elif fraud_votes >= len(predictions) // 2 + 1:
            confidence = "MEDIUM"
        elif fraud_votes == 1:
            confidence = "LOW"
        else:
            confidence = "VERY_LOW"
        
        # Generate reasoning
        reasoning = get_fraud_reasoning(transaction, features, predictions, scores)
        
        # Calculate prediction time
        prediction_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Log prediction
        logger.info(
            f"Transaction {transaction.transaction_id}: "
            f"Fraud={is_fraud}, Probability={fraud_probability:.4f}, "
            f"Confidence={confidence}, Time={prediction_time:.2f}ms"
        )
        
        # Check latency requirement
        if prediction_time > 100:
            logger.warning(f"Prediction time {prediction_time:.2f}ms exceeds 100ms target!")
        
        # Prepare response
        response = PredictionResponse(
            transaction_id=transaction.transaction_id,
            is_fraud=is_fraud,
            fraud_probability=round(fraud_probability, 4),
            confidence=confidence,
            reasoning=reasoning,
            models_agreement=predictions,
            prediction_time_ms=round(prediction_time, 2),
            timestamp=datetime.utcnow().isoformat()
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing transaction {transaction.transaction_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "timestamp": datetime.utcnow().isoformat()
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")