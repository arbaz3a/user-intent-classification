"""
main.py — FastAPI application for the ML prediction service

Exposes a POST /predict endpoint that receives a customer support message
and returns the predicted intent, confidence score, and processing time.
Also includes a GET /health endpoint for health checks.

Runs on port 8000.
"""

import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from predict import predict_intent

# ============================================================
# FastAPI app setup
# ============================================================

app = FastAPI(
    title="User Intent Classification API",
    description="ML-powered customer support intent classifier",
    version="1.0.0"
)

# ============================================================
# CORS middleware — allow all origins for development
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],         # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],         # Allow all HTTP methods
    allow_headers=["*"],         # Allow all headers
)


# ============================================================
# Pydantic models for request/response validation
# ============================================================

class PredictRequest(BaseModel):
    """Request body for the /predict endpoint."""
    message: str


class PredictResponse(BaseModel):
    """Response body for the /predict endpoint."""
    intent: str
    confidence: float
    processing_time_ms: float


# ============================================================
# Endpoints
# ============================================================

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Receives a customer support message and returns the predicted intent.

    - Cleans the text using the same pipeline as training
    - Transforms with the fitted TF-IDF vectorizer
    - Predicts with the trained model
    - Returns intent, confidence, and processing time in ms
    """
    print(f"[main.py] Received prediction request: '{request.message}'")

    # Measure processing time
    start_time = time.time()

    # Call the prediction function
    result = predict_intent(request.message)

    # Calculate processing time in milliseconds
    end_time = time.time()
    processing_time_ms = round((end_time - start_time) * 1000, 2)

    print(f"[main.py] Prediction complete in {processing_time_ms}ms")

    return PredictResponse(
        intent=result["intent"],
        confidence=round(result["confidence"], 4),
        processing_time_ms=processing_time_ms
    )


@app.get("/health")
async def health_check():
    """Health check endpoint to verify the API is running."""
    return {"status": "ok", "service": "ml-api"}


# ============================================================
# Run with uvicorn when executed directly
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
