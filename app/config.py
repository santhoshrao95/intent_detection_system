"""
Configuration settings for Intent Detection API
==============================================

Simple configuration management using Pydantic Settings.
Loads settings from environment variables or uses defaults.

Author: Intent Detection System
Date: 2025
"""

try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings

from typing import Optional
import os


class Settings(BaseSettings):
    """
    Application configuration settings.
    
    Settings can be overridden by environment variables.
    Example: export MODEL_PATH="/path/to/model.pkl"
    """
    
    # Model Configuration
    MODEL_PATH: str = "models/best_model.pkl"
    PIPELINE_PATH: str = "models/best_pipeline.pkl"
    CONFIDENCE_THRESHOLD: float = 0.75
    TOP_K_CLASSES: int = 5
    
    # OpenAI Configuration
    OPENAI_API_KEY: str = "your-openai-api-key-here"
    OPENAI_MODEL: str = "gpt-3.5-turbo"
    OPENAI_MAX_TOKENS: int = 150
    OPENAI_TEMPERATURE: float = 0.1
    
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_DEBUG: bool = False
    
    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
    
    def __init__(self, **kwargs):
        """Initialize settings with validation."""
        super().__init__(**kwargs)
        
        # Validate OpenAI API key
        if self.OPENAI_API_KEY == "your-openai-api-key-here":
            print("⚠️  WARNING: Please set your OpenAI API key!")
            print("   Set environment variable: export OPENAI_API_KEY='your-key'")
            print("   Or create .env file with: OPENAI_API_KEY=your-key")
        
        # Validate model paths
        if not os.path.exists(self.MODEL_PATH):
            print(f"⚠️  WARNING: Model file not found: {self.MODEL_PATH}")
            print("   Make sure you have trained a model first!")
        
        if not os.path.exists(self.PIPELINE_PATH):
            print(f"⚠️  WARNING: Pipeline file not found: {self.PIPELINE_PATH}")
            print("   Make sure you have trained a model first!")
    
    def get_openai_config(self) -> dict:
        """Get OpenAI configuration as dictionary."""
        return {
            "api_key": self.OPENAI_API_KEY,
            "model": self.OPENAI_MODEL,
            "max_tokens": self.OPENAI_MAX_TOKENS,
            "temperature": self.OPENAI_TEMPERATURE
        }
    
    def get_model_config(self) -> dict:
        """Get model configuration as dictionary."""
        return {
            "model_path": self.MODEL_PATH,
            "pipeline_path": self.PIPELINE_PATH,
            "confidence_threshold": self.CONFIDENCE_THRESHOLD,
            "top_k_classes": self.TOP_K_CLASSES
        }


# Create global settings instance
settings = Settings()


# Intent descriptions for LLM prompts
INTENT_DESCRIPTIONS = {
    "EMI": "Questions, requests, or inquiries about installment payments, EMI options, or payment plans",
    "COD": "Questions, requests, or inquiries about cash on delivery payment option",
    "ORTHO_FEATURES": "Questions, requests, or inquiries about orthopedic mattress features and benefits",
    "ERGO_FEATURES": "Questions, requests, or inquiries about ergonomic mattress features and benefits", 
    "COMPARISON": "Questions, requests, or inquiries comparing different mattress types or products",
    "100_NIGHT_TRIAL_OFFER": "Questions, requests, or inquiries about the 100-night trial period and return policy during trial",
    "SIZE_CUSTOMIZATION": "Questions, requests, or inquiries about customizing mattress size or dimensions",
    "WHAT_SIZE_TO_ORDER": "Questions, requests, or inquiries about what mattress size to choose or order",
    "LEAD_GEN": "General inquiries, requests, or needing personal assistance or wanting to talk to someone",
    "CHECK_PINCODE": "Questions, requests, or inquiries about delivery availability to specific locations or pincodes",
    "DISTRIBUTORS": "Questions, requests, or inquiries about physical stores, showrooms, or retail locations",
    "MATTRESS_COST": "Questions, requests, or inquiries about mattress pricing, cost, or how much products cost",
    "PRODUCT_VARIANTS": "Questions, requests, or inquiries about different mattress types, variants, or product options",
    "ABOUT_SOF_MATTRESS": "Questions, requests, or inquiries about the company, brand, or general information about SOF mattresses",
    "DELAY_IN_DELIVERY": "Complaints, requests, or inquiries about delayed or late deliveries",
    "ORDER_STATUS": "Questions, requests, or inquiries about checking order status or tracking orders",
    "RETURN_EXCHANGE": "Questions, requests, or inquiries about returning, exchanging, or getting refunds for products",
    "CANCEL_ORDER": "Requests to cancel orders or questions about cancellation process",
    "PILLOWS": "Questions, requests, or inquiries about pillow products, availability, or purchasing pillows",
    "OFFERS": "Questions, requests, or inquiries about discounts, offers, deals, or promotional pricing"
}


# Example queries for each intent (for LLM context)
INTENT_EXAMPLES = {
    "EMI": ["Do you provide EMI options?", "I want installment payment"],
    "COD": ["Is COD available?", "Can I pay cash on delivery?"],
    "ORTHO_FEATURES": ["What are ortho mattress features?", "Tell me about orthopedic benefits"],
    "ERGO_FEATURES": ["What are ergo mattress features?", "Ergonomic mattress benefits"],
    "COMPARISON": ["Difference between ortho and ergo?", "Compare your mattresses"],
    "WARRANTY": ["What is warranty period?", "Does warranty cover mattress cover?"],
    "100_NIGHT_TRIAL_OFFER": ["How does 100 night trial work?", "What is trial period?"],
    "SIZE_CUSTOMIZATION": ["Can I customize mattress size?", "I need custom dimensions"],
    "WHAT_SIZE_TO_ORDER": ["What size should I order?", "How to know bed size?"],
    "LEAD_GEN": ["I want to talk to someone", "Get in touch with agent"],
    "CHECK_PINCODE": ["Do you deliver to my area?", "Check delivery for pincode"],
    "DISTRIBUTORS": ["Do you have showrooms?", "Where can I see the product?"],
    "MATTRESS_COST": ["What is the price?", "How much does mattress cost?"],
    "PRODUCT_VARIANTS": ["What mattress types do you have?", "Show me product options"],
    "ABOUT_SOF_MATTRESS": ["Tell me about SOF mattress", "What is your company?"],
    "DELAY_IN_DELIVERY": ["My order is delayed", "When will I get my mattress?"],
    "ORDER_STATUS": ["What is my order status?", "Track my order"],
    "RETURN_EXCHANGE": ["I want to return", "How to exchange mattress?"],
    "CANCEL_ORDER": ["Cancel my order", "I want to cancel purchase"],
    "PILLOWS": ["Do you sell pillows?", "I want to buy pillows"],
    "OFFERS": ["Any discounts available?", "What offers do you have?"]
}


if __name__ == "__main__":
    # Test configuration loading
    print("🔧 Configuration Test")
    print("=" * 30)
    print(f"Model Path: {settings.MODEL_PATH}")
    print(f"Pipeline Path: {settings.PIPELINE_PATH}")
    print(f"Confidence Threshold: {settings.CONFIDENCE_THRESHOLD}")
    print(f"OpenAI Model: {settings.OPENAI_MODEL}")
    print(f"API Port: {settings.API_PORT}")
    print("\n✅ Configuration loaded successfully!")