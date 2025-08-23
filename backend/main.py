from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timedelta
import httpx
import math
import pickle
import os
import json
from pathlib import Path

app = FastAPI(title="McDaniels Container Shipping API", version="1.0.0")

# CORS for Svelte frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Default warehouse location (Lexington, KY)
WAREHOUSE_LAT = 38.0406
WAREHOUSE_LON = -84.5037
WAREHOUSE_ZIP = "40507"

class QuoteRequest(BaseModel):
    origin_zip: str = WAREHOUSE_ZIP
    dest_zip: str
    container_size: str = "20ft"  # "20ft" or "40ft"

class QuoteResponse(BaseModel):
    price: float
    dest_state: str
    distance_miles: float
    eta_window: str
    timestamp: str
    model_used: bool
    features: dict
    confidence: Optional[float] = None

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points using Haversine formula"""
    R = 3959  # Earth's radius in miles
    
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    return R * c

async def geocode_zip(zip_code: str) -> tuple[float, float, str]:
    """Get lat/lon and state for ZIP code using Zippopotam.us"""
    url = f"https://api.zippopotam.us/us/{zip_code}"
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.get(url)
            if response.status_code != 200:
                raise HTTPException(status_code=400, detail=f"Invalid ZIP code: {zip_code}")
            
            data = response.json()
            place = data["places"][0]
            
            return (
                float(place["latitude"]),
                float(place["longitude"]),
                place["state abbreviation"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Geocoding failed: {str(e)}")

def build_time_features() -> dict:
    """Build time-aware features for ML model"""
    now = datetime.now()
    
    return {
        "minute_of_day": now.hour * 60 + now.minute,
        "day_of_week": now.weekday(),  # 0 = Monday
        "is_weekend": 1 if now.weekday() >= 5 else 0,
        "hour": now.hour,
        "month": now.month,
        "quarter": (now.month - 1) // 3 + 1
    }

def fallback_pricing(distance_miles: float, container_size: str, time_features: dict) -> float:
    """Fallback pricing formula when ML model is not available"""
    
    # Base rates
    base_rates = {
        "20ft": 2.10,  # per mile
        "40ft": 2.85   # per mile
    }
    
    base_rate = base_rates.get(container_size, base_rates["20ft"])
    
    # Distance-based pricing with economies of scale
    if distance_miles <= 100:
        rate_multiplier = 1.4  # Higher rate for short distances
    elif distance_miles <= 300:
        rate_multiplier = 1.0
    elif distance_miles <= 600:
        rate_multiplier = 0.85
    else:
        rate_multiplier = 0.75  # Best rate for long distances
    
    # Time-based adjustments
    time_multiplier = 1.0
    
    # Weekend surcharge
    if time_features["is_weekend"]:
        time_multiplier += 0.15
    
    # Peak hours (7-9 AM, 4-6 PM)
    hour = time_features["hour"]
    if (7 <= hour <= 9) or (16 <= hour <= 18):
        time_multiplier += 0.10
    
    # Holiday season (Q4)
    if time_features["quarter"] == 4:
        time_multiplier += 0.08
    
    # Calculate price
    base_price = distance_miles * base_rate * rate_multiplier
    adjusted_price = base_price * time_multiplier
    
    # Minimum price
    min_prices = {"20ft": 450, "40ft": 600}
    final_price = max(adjusted_price, min_prices.get(container_size, 450))
    
    return round(final_price, 2)

def load_ml_model():
    """Load ML model if available"""
    model_path = Path("model/model.pkl")
    if model_path.exists():
        try:
            with open(model_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Failed to load ML model: {e}")
    return None

def predict_with_model(model, features: dict) -> tuple[float, float]:
    """Make prediction with ML model"""
    import pandas as pd
    
    # Convert features to DataFrame
    feature_names = [
        'distance_miles', 'container_size_encoded', 'minute_of_day', 
        'day_of_week', 'is_weekend', 'hour', 'month', 'quarter'
    ]
    
    df = pd.DataFrame([features])
    df = df.reindex(columns=feature_names, fill_value=0)
    
    prediction = model.predict(df)[0]
    
    # Get confidence if model supports it
    confidence = None
    if hasattr(model, 'predict_proba'):
        try:
            proba = model.predict_proba(df)[0]
            confidence = float(max(proba))
        except:
            pass
    
    return float(prediction), confidence

def calculate_eta(distance_miles: float, time_features: dict) -> str:
    """Calculate ETA window based on distance and time factors"""
    
    # Base delivery times (business days)
    if distance_miles <= 150:
        base_days = 1
        window_days = 1  # 1-2 days
    elif distance_miles <= 400:
        base_days = 2
        window_days = 2  # 2-4 days
    elif distance_miles <= 800:
        base_days = 3
        window_days = 2  # 3-5 days
    else:
        base_days = 5
        window_days = 3  # 5-8 days
    
    # Weekend adjustments
    if time_features["is_weekend"]:
        base_days += 1
        window_days += 1
    
    # Holiday season delays
    if time_features["quarter"] == 4:
        base_days += 1
        window_days += 1
    
    min_days = max(1, base_days)
    max_days = min_days + window_days
    
    return f"{min_days}-{max_days} business days"

# Load ML model on startup
ml_model = load_ml_model()
if ml_model:
    print("✅ ML model loaded successfully")
else:
    print("⚠️  No ML model found, using fallback pricing")

@app.get("/")
async def root():
    return {
        "message": "McDaniels Container Shipping API",
        "version": "1.0.0",
        "ml_model_loaded": ml_model is not None
    }

@app.post("/quote", response_model=QuoteResponse)
async def get_quote(request: QuoteRequest):
    """Get shipping quote with ML pricing or fallback formula"""
    
    # Geocode destination
    dest_lat, dest_lon, dest_state = await geocode_zip(request.dest_zip)
    
    # Calculate distance
    distance_miles = haversine_distance(WAREHOUSE_LAT, WAREHOUSE_LON, dest_lat, dest_lon)
    
    # Build features
    time_features = build_time_features()
    
    # Container size encoding for ML
    container_size_encoded = 1 if request.container_size == "40ft" else 0
    
    features = {
        "distance_miles": distance_miles,
        "container_size_encoded": container_size_encoded,
        **time_features
    }
    
    # Get price prediction
    model_used = False
    confidence = None
    
    if ml_model is not None:
        try:
            price, confidence = predict_with_model(ml_model, features)
            model_used = True
        except Exception as e:
            print(f"ML prediction failed: {e}, using fallback")
            price = fallback_pricing(distance_miles, request.container_size, time_features)
    else:
        price = fallback_pricing(distance_miles, request.container_size, time_features)
    
    # Calculate ETA
    eta_window = calculate_eta(distance_miles, time_features)
    
    return QuoteResponse(
        price=price,
        dest_state=dest_state,
        distance_miles=round(distance_miles, 1),
        eta_window=eta_window,
        timestamp=datetime.now().isoformat(),
        model_used=model_used,
        features=features,
        confidence=confidence
    )

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "ml_model_available": ml_model is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
