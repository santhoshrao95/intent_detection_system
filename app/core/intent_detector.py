"""
Core Intent Detection System
===========================

Main orchestrator for the two-phase intent detection:
1. High confidence (>0.75): Direct ML prediction (fast path)
2. Low confidence (≤0.75): LLM-assisted reasoning (smart path)

Author: Intent Detection System
Date: 2025
"""

import numpy as np
import joblib
import pickle
import time
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from app.core.llm_handler import LLMHandler

logger = logging.getLogger(__name__)


class IntentDetectionSystem:
    """
    Main intent detection system that combines ML and LLM approaches.
    
    Flow:
    1. Preprocess query using existing pipeline
    2. Get ML prediction + probabilities  
    3. If max probability > threshold: return ML result (fast path)
    4. If max probability ≤ threshold: use LLM with top-k classes (smart path)
    """
    
    def __init__(
        self, 
        model_path: str,
        pipeline_path: str, 
        confidence_threshold: float = 0.75,
        openai_api_key: str = "",
        openai_model: str = "gpt-3.5-turbo"
    ):
        """
        Initialize the intent detection system.
        
        Args:
            model_path: Path to trained ML model (pickle file)
            pipeline_path: Path to data preprocessing pipeline (pickle file)
            confidence_threshold: Threshold for high/low confidence routing
            openai_api_key: OpenAI API key for LLM fallback
            openai_model: OpenAI model name to use
        """
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.pipeline = None
        self.llm_handler = None
        
        # Load ML model and pipeline
        self._load_model(model_path, pipeline_path)
        
        # Initialize LLM handler
        if openai_api_key and openai_api_key != "your-openai-api-key-here":
            self.llm_handler = LLMHandler(
                api_key=openai_api_key,
                model=openai_model
            )
            logger.info("✅ LLM handler initialized")
        else:
            logger.warning("⚠️ LLM handler not initialized - OpenAI API key not provided")
            logger.warning("   Low confidence queries will fallback to ML prediction")
    
    def _load_model(self, model_path: str, pipeline_path: str):
        """Load the trained ML model and preprocessing pipeline."""
        try:
            # Load the trained model
            logger.info(f"Loading ML model from: {model_path}")
            self.model = joblib.load(model_path)
            logger.info(f"✅ Model loaded: {type(self.model).__name__}")
            
            # Load the preprocessing pipeline  
            logger.info(f"Loading pipeline from: {pipeline_path}")
            with open(pipeline_path, 'rb') as f:
                self.pipeline = pickle.load(f)
            logger.info(f"✅ Pipeline loaded: {type(self.pipeline).__name__}")
            
            # Verify model has required methods
            if not hasattr(self.model, 'predict_proba'):
                raise ValueError("Model must support predict_proba for confidence scoring")
                
            # Get class names
            if hasattr(self.model, 'classes_'):
                self.class_names = self.model.classes_
                logger.info(f"✅ Found {len(self.class_names)} classes: {list(self.class_names[:5])}...")
            else:
                raise ValueError("Model must have classes_ attribute")
                
        except Exception as e:
            logger.error(f"❌ Failed to load model/pipeline: {str(e)}")
            raise
    
    async def detect_intent(self, query: str) -> Dict:
        """
        Main intent detection method.
        
        Args:
            query: User input text
            
        Returns:
            Dictionary with intent, confidence, processing_path, etc.
        """
        start_time = time.time()
        
        try:
            # Get ML prediction and probabilities
            predicted_intent, probabilities = self._get_ml_prediction(query)
            max_confidence = np.max(probabilities)
            
            logger.info(f"ML prediction: {predicted_intent} (confidence: {max_confidence:.3f})")
            
            # Route based on confidence
            if max_confidence > self.confidence_threshold:
                # High confidence - fast path
                result = self._fast_path_result(
                    query, predicted_intent, max_confidence, probabilities
                )
                processing_path = "fast"
                logger.info("🚀 Using fast path (high confidence)")
                
            else:
                # Low confidence - smart path with LLM
                if self.llm_handler:
                    result = await self._smart_path_result(
                        query, predicted_intent, max_confidence, probabilities
                    )
                    processing_path = "smart"
                    logger.info("🧠 Using smart path (LLM assisted)")
                else:
                    # Fallback to ML if no LLM available
                    result = self._fast_path_result(
                        query, predicted_intent, max_confidence, probabilities
                    )
                    processing_path = "fast"
                    logger.warning("⚠️ LLM not available, using ML fallback")
            
            # Calculate response time
            response_time = (time.time() - start_time) * 1000  # Convert to ms
            
            # Build final result
            final_result = {
                "intent": result["intent"],
                "confidence": result["confidence"], 
                "processing_path": processing_path,
                "response_time_ms": round(response_time, 1),
                "reasoning": result.get("reasoning"),
                "alternatives": result.get("alternatives")
            }
            
            return final_result
            
        except Exception as e:
            logger.error(f"❌ Error in intent detection: {str(e)}")
            raise
    
    def _get_ml_prediction(self, query: str) -> Tuple[str, np.ndarray]:
        """
        Get ML model prediction and probabilities.
        
        Args:
            query: User input text
            
        Returns:
            Tuple of (predicted_class, probabilities_array)
        """
        try:
            # Transform query using the preprocessing pipeline
            X = self.pipeline.transform([query])
            
            # Get prediction and probabilities
            prediction = self.model.predict(X)[0]
            probabilities = self.model.predict_proba(X)[0]
            
            return prediction, probabilities
            
        except Exception as e:
            logger.error(f"❌ Error in ML prediction: {str(e)}")
            raise
    
    def _get_top_k_classes(self, probabilities: np.ndarray, k: int = 5) -> List[Dict]:
        """
        Get top-k classes with their probabilities.
        
        Args:
            probabilities: Array of class probabilities
            k: Number of top classes to return
            
        Returns:
            List of dicts with intent and confidence
        """
        # Get top k indices
        top_k_indices = np.argsort(probabilities)[-k:][::-1]
        
        top_k_classes = []
        for idx in top_k_indices:
            top_k_classes.append({
                "intent": self.class_names[idx],
                "confidence": float(probabilities[idx])
            })
        
        return top_k_classes
    
    def _fast_path_result(
        self, 
        query: str, 
        predicted_intent: str, 
        confidence: float,
        probabilities: np.ndarray
    ) -> Dict:
        """
        Generate result for fast path (high confidence ML prediction).
        
        Args:
            query: Original query
            predicted_intent: ML predicted intent
            confidence: Confidence score
            probabilities: All class probabilities
            
        Returns:
            Result dictionary
        """
        return {
            "intent": predicted_intent,
            "confidence": float(confidence),
            "reasoning": None,
            "alternatives": None
        }
    
    async def _smart_path_result(
        self,
        query: str,
        ml_predicted_intent: str, 
        ml_confidence: float,
        probabilities: np.ndarray
    ) -> Dict:
        """
        Generate result for smart path (LLM-assisted reasoning).
        
        Args:
            query: Original query
            ml_predicted_intent: ML predicted intent  
            ml_confidence: ML confidence score
            probabilities: All class probabilities
            
        Returns:
            Result dictionary with LLM reasoning
        """
        try:
            # Get top 5 classes for LLM consideration
            top_k_classes = self._get_top_k_classes(probabilities, k=5)
            
            # Use LLM to classify with reasoning
            llm_result = await self.llm_handler.classify_with_llm(query, top_k_classes)
            
            # Prepare alternatives (top classes excluding the selected one)
            alternatives = [
                cls for cls in top_k_classes 
                if cls["intent"] != llm_result["intent"]
            ][:3]  # Show top 3 alternatives
            
            return {
                "intent": llm_result["intent"],
                "confidence": llm_result["confidence"],
                "reasoning": llm_result["reasoning"],
                "alternatives": alternatives if alternatives else None
            }
            
        except Exception as e:
            logger.error(f"❌ Error in smart path: {str(e)}")
            # Fallback to ML prediction if LLM fails
            logger.warning("🔄 Falling back to ML prediction due to LLM error")
            return {
                "intent": ml_predicted_intent,
                "confidence": float(ml_confidence),
                "reasoning": f"LLM fallback due to error: {str(e)}",
                "alternatives": None
            }
    
    def get_confidence_distribution(self, query: str) -> Dict[str, float]:
        """
        Get confidence scores for all classes (for debugging/analysis).
        
        Args:
            query: User input text
            
        Returns:
            Dictionary mapping class names to confidence scores
        """
        try:
            _, probabilities = self._get_ml_prediction(query)
            
            confidence_dist = {}
            for i, class_name in enumerate(self.class_names):
                confidence_dist[class_name] = float(probabilities[i])
            
            # Sort by confidence (descending)
            return dict(sorted(confidence_dist.items(), key=lambda x: x[1], reverse=True))
            
        except Exception as e:
            logger.error(f"❌ Error getting confidence distribution: {str(e)}")
            return {}
    
    def health_check(self) -> bool:
        """
        Perform a health check on the intent detection system.
        
        Returns:
            True if system is healthy, False otherwise
        """
        try:
            # Test with a simple query
            test_query = "test health check"
            _, probabilities = self._get_ml_prediction(test_query)
            
            # Verify probabilities sum to approximately 1
            prob_sum = np.sum(probabilities)
            if not (0.99 <= prob_sum <= 1.01):
                logger.error(f"❌ Health check failed: probabilities sum to {prob_sum}")
                return False
            
            logger.info("✅ Health check passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Health check failed: {str(e)}")
            return False


if __name__ == "__main__":
    # Test the intent detection system
    print("🧪 Testing Intent Detection System")
    print("=" * 40)
    
    # This would be used for testing during development
    # detector = IntentDetectionSystem(
    #     model_path="../models/best_model.pkl",
    #     pipeline_path="../models/best_pipeline.pkl"
    # )
    
    print("✅ Intent Detection System module loaded successfully!")