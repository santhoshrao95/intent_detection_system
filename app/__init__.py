"""
Intent Detection FastAPI Application
===================================

A production-ready API for intent detection using a hybrid approach:
- High confidence queries: Fast ML-only prediction
- Low confidence queries: Smart ML + LLM reasoning

Author: Intent Detection System
Date: 2025
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Intent Detection System"
__description__ = "FastAPI application for intelligent intent detection"

# Package metadata
__all__ = [
    "__version__",
    "__author__", 
    "__description__"
]

# Import main components for easy access
try:
    from app.core.intent_detector import IntentDetectionSystem
    from app.core.llm_handler import LLMHandler
    from app.models.schemas import (
        IntentRequest,
        IntentResponse, 
        HealthResponse,
        AlternativeIntent,
        ErrorResponse
    )
    from app.config import Settings, INTENT_DESCRIPTIONS, INTENT_EXAMPLES
    
    # Add to __all__ if imports successful
    __all__.extend([
        "IntentDetectionSystem",
        "LLMHandler", 
        "IntentRequest",
        "IntentResponse",
        "HealthResponse",
        "AlternativeIntent", 
        "ErrorResponse",
        "Settings",
        "INTENT_DESCRIPTIONS",
        "INTENT_EXAMPLES"
    ])
    
except ImportError as e:
    # Don't fail package import if dependencies aren't available
    import warnings
    warnings.warn(f"Some imports failed: {e}")

# Package-level configuration
DEFAULT_CONFIG = {
    "confidence_threshold": 0.75,
    "top_k_classes": 5,
    "api_version": "v1",
    "model_type": "sentence_bert + logistic_regression"
}

def get_version() -> str:
    """Get package version."""
    return __version__

def get_config() -> dict:
    """Get default configuration."""
    return DEFAULT_CONFIG.copy()

def print_info():
    """Print package information."""
    print(f"""
🤖 Intent Detection API
=====================
Version: {__version__}
Author: {__author__}
Description: {__description__}

Features:
✅ Hybrid ML + LLM approach
✅ SentenceBERT + Logistic Regression
✅ OpenAI integration for uncertain cases  
✅ FastAPI with automatic documentation
✅ Production-ready error handling
""")

if __name__ == "__main__":
    print_info()