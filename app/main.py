import logging
from contextlib import asynccontextmanager
from typing import Dict
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class PredictionRequest(BaseModel):
    """Request model for sentiment prediction."""
    text: str = Field(..., min_length=1, description="User input text for sentiment analysis.")

class PredictionResponse(BaseModel):
    """Response model for sentiment prediction."""
    prediction: str = Field(..., description="Predictio label, e.g., 'POSITIVE' or 'NEGATIVE'.")
    probability: float = Field(..., description="Probability of the predicted sentiment, between 0 and 1.")


def load_model_pipeline():
    """
    Loads the Hugging Face tokenizer and model at service startup.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model=model_name,
        device=device
    )
    logging.info(f"Model '{model_name}' loaded successfully on device '{device}'.")
    return sentiment_pipeline

lifespan_context: Dict = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan manager to load the model on startup.
    """
    logging.info("Service startup: Loading model...")
    lifespan_context["sentiment_pipeline"] = load_model_pipeline()
    yield
    logging.info("Service shutdown: Clearing resources...")
    lifespan_context.clear()


app = FastAPI(
    title="English Sentiment Analysis Service",
    description="A microservice to predict sentiment from text using a Hugging Face model.",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/", tags=["Health Check"])
def read_root():
    """Healthcheck endpoint to verify service status."""
    return {"status": "ok", "message": "Service is running."}

@app.post("/predict_sentiment", response_model=PredictionResponse, tags=["Predictions"])
def predict_sentiment(request: PredictionRequest) -> PredictionResponse:
    """
    Receives text and returns the predicted sentiment and its probability.
    """
    sentiment_pipeline = lifespan_context.get("sentiment_pipeline")
    if not sentiment_pipeline:
        logging.error("Model not loaded or unavailable.")
        raise HTTPException(status_code=503, detail="Model not loaded or unavailable.")

    result = sentiment_pipeline(request.text, top_k=1)[0]
    logging.info(f"Prediction successful for text: '{request.text[:30]}...'")

    return PredictionResponse(
        prediction=result['label'],
        probability=round(result['score'], 4)
    )