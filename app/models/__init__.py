"""
Pydantic Models Package
======================

Contains all Pydantic models for API request/response schemas.
Provides type safety and automatic validation for the FastAPI endpoints.

Author: Intent Detection System
Date: 2025
"""

# Import all schemas for easy access
from app.models.schemas import (
    IntentRequest,
    IntentResponse,
    AlternativeIntent,
    HealthResponse,
    ErrorResponse
)

__all__ = [
    "IntentRequest",
    "IntentResponse", 
    "AlternativeIntent",
    "HealthResponse",
    "ErrorResponse"
]

# Schema information for documentation
SCHEMA_INFO = {
    "IntentRequest": {
        "description": "Request model for intent detection",
        "required_fields": ["query"],
        "validation": "Query length 1-500 characters"
    },
    "IntentResponse": {
        "description": "Response model with intent and confidence",
        "fields": ["intent", "confidence", "processing_path", "response_time_ms"],
        "optional_fields": ["reasoning", "alternatives"]
    },
    "AlternativeIntent": {
        "description": "Alternative intent option with confidence",
        "fields": ["intent", "confidence"]
    },
    "HealthResponse": {
        "description": "System health status",
        "fields": ["status", "model_loaded", "timestamp", "version"]
    },
    "ErrorResponse": {
        "description": "Standardized error response",
        "fields": ["error", "message", "details", "timestamp"]
    }
}

def get_schema_info() -> dict:
    """Get information about all schemas."""
    return SCHEMA_INFO.copy()

def list_schemas() -> list:
    """Get list of available schema names."""
    return list(__all__)