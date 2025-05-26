"""
Core Business Logic Package
===========================

Contains the main business logic components for intent detection:
- IntentDetectionSystem: Main orchestrator
- LLMHandler: OpenAI integration for low-confidence cases

Author: Intent Detection System
Date: 2025
"""

# Import core components
from app.core.intent_detector import IntentDetectionSystem
from app.core.llm_handler import LLMHandler

__all__ = [
    "IntentDetectionSystem",
    "LLMHandler"
]

# Core component information
CORE_INFO = {
    "IntentDetectionSystem": {
        "description": "Main intent detection orchestrator",
        "features": [
            "Two-path routing (fast/smart)",
            "ML model integration", 
            "Confidence-based decision making",
            "Health monitoring"
        ],
        "dependencies": ["ML model", "preprocessing pipeline"]
    },
    "LLMHandler": {
        "description": "LLM integration for uncertain cases",
        "features": [
            "OpenAI API integration",
            "Structured prompt generation",
            "Response parsing and validation",
            "Error handling and fallbacks"
        ],
        "dependencies": ["OpenAI API key"]
    }
}

def get_core_info() -> dict:
    """Get information about core components."""
    return CORE_INFO.copy()

def list_components() -> list:
    """Get list of available core components."""
    return list(__all__)

# System flow information
SYSTEM_FLOW = {
    "fast_path": {
        "trigger": "ML confidence > threshold",
        "components": ["IntentDetectionSystem", "ML Model"],
        "response_time": "~100ms",
        "accuracy": "High for clear intents"
    },
    "smart_path": {
        "trigger": "ML confidence ≤ threshold", 
        "components": ["IntentDetectionSystem", "LLMHandler", "OpenAI API"],
        "response_time": "~2-3 seconds",
        "accuracy": "Higher for ambiguous intents"
    }
}

def get_system_flow() -> dict:
    """Get information about system processing flow."""
    return SYSTEM_FLOW.copy()