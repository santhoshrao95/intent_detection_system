"""
FastAPI application for Intent Detection System
=============================================

Simple API with two main endpoints:
- POST /detect: Main intent detection endpoint
- GET /health: Health check endpoint

Author: Intent Detection System
Date: 2025
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
import sys
from pathlib import Path

# Add utils to path for importing ML pipeline
sys.path.append(str(Path(__file__).parent.parent))

from app.models.schemas import IntentRequest, IntentResponse, HealthResponse
from app.core.intent_detector import IntentDetectionSystem
from app.config import Settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Intent Detection API",
    description="AI-powered intent detection system with ML + LLM fallback",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load configuration
settings = Settings()

# Global variable for intent detection system
intent_detector = None


@app.on_event("startup")
async def startup_event():
    """Initialize the intent detection system on startup."""
    global intent_detector

    try:
        logger.info("🚀 Starting Intent Detection API...")
        logger.info("Loading ML model and preprocessing pipeline...")

        intent_detector = IntentDetectionSystem(
            model_path=settings.MODEL_PATH,
            pipeline_path=settings.PIPELINE_PATH,
            confidence_threshold=settings.CONFIDENCE_THRESHOLD,
            openai_api_key=settings.OPENAI_API_KEY,
            openai_model=settings.OPENAI_MODEL,
        )

        logger.info("✅ Intent Detection System initialized successfully!")

    except Exception as e:
        logger.error(f"❌ Failed to initialize Intent Detection System: {str(e)}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("🛑 Shutting down Intent Detection API...")


@app.post("/detect", response_model=IntentResponse)
async def detect_intent(request: IntentRequest):
    """
    Detect intent from user query.

    Uses two-path approach:
    - High confidence (>0.75): Direct ML prediction
    - Low confidence (<0.75): LLM-assisted reasoning
    """
    try:
        if not intent_detector:
            raise HTTPException(
                status_code=503, detail="Intent detection system not initialized"
            )

        logger.info(f"Processing query: '{request.query}'")

        # Detect intent using the two-path system
        result = await intent_detector.detect_intent(request.query)

        logger.info(f"Result: {result['intent']} ({result['processing_path']} path)")

        return IntentResponse(**result)

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error processing intent detection: {str(e)}"
        )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns the status of the API and its dependencies.
    """
    try:
        model_loaded = intent_detector is not None

        if model_loaded:
            # Test model with a simple query
            test_result = await intent_detector.detect_intent("test")
            status = "healthy"
        else:
            status = "unhealthy"

        return HealthResponse(status=status, model_loaded=model_loaded)

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(status="unhealthy", model_loaded=False)


@app.get("/")
async def root():
    """Root endpoint with basic API information."""
    return {
        "message": "Intent Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


