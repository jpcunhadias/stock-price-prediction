from fastapi import FastAPI, HTTPException
from .schemas import PredictionRequest
from .model_loader import predict_price, get_last_60_days_prices

app = FastAPI(
    title="Stock Price Prediction API",
    description="A RESTful API to predict stock prices from Petrobras (PETR4.SA) using LSTM models.",
    version="1.0.0",
)


@app.get("/health", status_code=200)
def health_check():
    """Check if the API is running."""
    return {"status": "ok", "message": "API is running"}


@app.get("/version", status_code=200)
def get_version():
    """Returns the API & model version."""
    return {"version": "1.0.0", "model_version": "latest"}


@app.get("/last_period/", status_code=200)
def fetch_last_period():
    """Fetches the last 60 days of stock prices for prediction."""
    last_60_days = get_last_60_days_prices()
    return {"history": last_60_days}


@app.post("/predict/")
def predict(request: PredictionRequest):
    """Handles prediction requests using LSTM model."""
    if len(request.history) != 60:
        raise HTTPException(
            status_code=400, detail="Input sequence must be exactly 60 days"
        )

    predicted_price = predict_price(request.history)

    return {
        "predicted_price": predicted_price,
        "unit": "R$",
        "message": "Prediction successful",
    }
