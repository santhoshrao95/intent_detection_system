"""
Pydantic models for API request/response schemas
==============================================

Defines the data models for API endpoints including
request validation and response formatting.

Author: Intent Detection System
Date: 2025
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Literal
from datetime import datetime


class IntentRequest(BaseModel):
    """
    Request model for intent detection endpoint.
    """
    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="User query text to classify",
        example="I want to cancel my order"
    )
    
    @validator('query')
    def validate_query(cls, v):
        """Validate and clean query text."""
        if not v or not v.strip():
            raise ValueError("Query cannot be empty")
        
        # Strip whitespace
        v = v.strip()
        
        # Basic length check
        if len(v) > 500:
            raise ValueError("Query too long (max 500 characters)")
            
        return v

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "query": "I want to cancel my order"
            }
        }


class IntentResponse(BaseModel):
    """
    Response model for intent detection endpoint.
    """
    intent: str = Field(
        ...,
        description="Detected intent class",
        example="CANCEL_ORDER"
    )
    
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0 and 1",
        example=0.89
    )
    
    processing_path: Literal["fast", "smart"] = Field(
        ...,
        description="Processing path used (fast=ML only, smart=ML+LLM)",
        example="fast"
    )
    
    response_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Processing time in milliseconds",
        example=156.7
    )
    
    reasoning: Optional[str] = Field(
        None,
        description="LLM reasoning (only for smart path)",
        example="The user explicitly mentions wanting to cancel their order, which clearly indicates a cancellation request."
    )
    
    alternatives: Optional[List['AlternativeIntent']] = Field(
        None,
        description="Alternative intents considered (only for smart path)",
        example=None
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "examples": [
                {
                    "description": "High confidence case (fast path)",
                    "value": {
                        "intent": "CANCEL_ORDER",
                        "confidence": 0.89,
                        "processing_path": "fast",
                        "response_time_ms": 156.7,
                        "reasoning": None,
                        "alternatives": None
                    }
                },
                {
                    "description": "Low confidence case (smart path)",
                    "value": {
                        "intent": "LEAD_GEN",
                        "confidence": 0.68,
                        "processing_path": "smart",
                        "response_time_ms": 2847.3,
                        "reasoning": "The query is general and asks for help, indicating the user needs assistance, which best matches the LEAD_GEN intent for connecting with support.",
                        "alternatives": [
                            {"intent": "WARRANTY", "confidence": 0.65},
                            {"intent": "ABOUT_SOF_MATTRESS", "confidence": 0.61}
                        ]
                    }
                }
            ]
        }


class AlternativeIntent(BaseModel):
    """
    Alternative intent option with confidence score.
    """
    intent: str = Field(
        ...,
        description="Alternative intent class",
        example="WARRANTY"
    )
    
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score for this alternative",
        example=0.65
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "intent": "WARRANTY",
                "confidence": 0.65
            }
        }


class HealthResponse(BaseModel):
    """
    Response model for health check endpoint.
    """
    status: Literal["healthy", "unhealthy"] = Field(
        ...,
        description="Overall system health status",
        example="healthy"
    )
    
    model_loaded: bool = Field(
        ...,
        description="Whether the ML model is loaded and ready",
        example=True
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp of health check",
        example="2025-01-15T10:30:00Z"
    )
    
    version: str = Field(
        default="1.0.0",
        description="API version",
        example="1.0.0"
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "examples": [
                {
                    "description": "Healthy system",
                    "value": {
                        "status": "healthy",
                        "model_loaded": True,
                        "timestamp": "2025-01-15T10:30:00Z",
                        "version": "1.0.0"
                    }
                },
                {
                    "description": "Unhealthy system",
                    "value": {
                        "status": "unhealthy",
                        "model_loaded": False,
                        "timestamp": "2025-01-15T10:30:00Z",
                        "version": "1.0.0"
                    }
                }
            ]
        }


class ErrorResponse(BaseModel):
    """
    Error response model for API errors.
    """
    error: str = Field(
        ...,
        description="Error type or code",
        example="ValidationError"
    )
    
    message: str = Field(
        ...,
        description="Human-readable error message",
        example="Query cannot be empty"
    )
    
    details: Optional[dict] = Field(
        None,
        description="Additional error details",
        example={"field": "query", "constraint": "min_length"}
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Error timestamp",
        example="2025-01-15T10:30:00Z"
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "error": "ValidationError", 
                "message": "Query cannot be empty",
                "details": {"field": "query", "constraint": "min_length"},
                "timestamp": "2025-01-15T10:30:00Z"
            }
        }


# Update forward references for nested models
IntentResponse.model_rebuild()