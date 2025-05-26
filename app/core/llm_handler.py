"""
LLM Handler for Intent Classification
====================================

Handles LLM integration for low-confidence cases using OpenAI API v1.0+.
Provides structured prompts and response parsing for intent classification.

Author: Intent Detection System
Date: 2025
"""

from openai import OpenAI
import logging
import json
import re
from typing import Dict, List, Optional
import asyncio

from app.config import INTENT_DESCRIPTIONS, INTENT_EXAMPLES

logger = logging.getLogger(__name__)


class LLMHandler:
    """
    Handles LLM-based intent classification for uncertain cases.
    
    Uses OpenAI API v1.0+ to reason about intent when ML model confidence is low.
    """
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        """
        Initialize LLM handler.
        
        Args:
            api_key: OpenAI API key
            model: OpenAI model name to use
        """
        self.api_key = api_key
        self.model = model
        
        # Configure OpenAI client (v1.0+ style)
        self.client = OpenAI(api_key=api_key)
        
        # LLM configuration
        self.max_tokens = 150
        self.temperature = 0.1  # Low temperature for consistent results
        
        logger.info(f"✅ LLM Handler initialized with model: {model}")
    
    async def classify_with_llm(self, query: str, top_classes: List[Dict]) -> Dict:
        """
        Classify intent using LLM reasoning.
        
        Args:
            query: User input text
            top_classes: List of top candidate classes with confidences
            
        Returns:
            Dictionary with intent, confidence, and reasoning
        """
        try:
            logger.info(f"🧠 LLM classification for query: '{query}'")
            logger.info(f"   Top candidates: {[cls['intent'] for cls in top_classes[:3]]}")
            
            # Build prompt with query and candidates
            prompt = self._build_prompt(query, top_classes)
            
            # Call OpenAI API
            response_text = await self._call_openai_api(prompt)
            
            # Parse LLM response
            result = self._parse_llm_response(response_text, top_classes)
            
            logger.info(f"✅ LLM result: {result['intent']} (confidence: {result['confidence']:.3f})")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in LLM classification: {str(e)}")
            raise
    
    def _build_prompt(self, query: str, top_classes: List[Dict]) -> str:
        """
        Build structured prompt for LLM intent classification.
        
        Args:
            query: User input text
            top_classes: Top candidate classes with confidences
            
        Returns:
            Formatted prompt string
        """
        # Build candidates section
        candidates_text = ""
        for i, cls in enumerate(top_classes, 1):
            intent = cls['intent']
            confidence = cls['confidence']
            description = INTENT_DESCRIPTIONS.get(intent, "No description available")
            examples = INTENT_EXAMPLES.get(intent, [])
            
            candidates_text += f"{i}. {intent} (ML confidence: {confidence:.3f})\n"
            candidates_text += f"   Description: {description}\n"
            
            if examples:
                examples_str = '", "'.join(examples[:2])  # Show 2 examples max
                candidates_text += f'   Examples: "{examples_str}"\n'
            
            candidates_text += "\n"
        
        # Build complete prompt
        prompt = f"""You are an expert intent classifier for a mattress company's customer service system.

USER QUERY: "{query}"

The ML model is uncertain about the intent. Here are the top 5 most likely candidates:

{candidates_text}

TASK: Analyze the user query carefully and determine which intent is most appropriate.

INSTRUCTIONS:
1. Consider the exact words and context in the user query
2. Match the query meaning to the intent descriptions and examples
3. Choose the intent that best captures what the user wants to accomplish
4. Provide a brief reasoning for your choice

RESPONSE FORMAT (respond exactly in this format):
Intent: [SELECTED_INTENT_NAME]
Confidence: [High/Medium/Low]
Reasoning: [Brief explanation of why this intent was chosen]

IMPORTANT: 
- DO NOT CONSIDER THE CONFIDENCE OF THE ML MODEL AT ALL. 
- MAKE YOUR OWN INFERENCE BASED ON THE USER QUERY AND THE INTENT DESCRIPTIONS AND EXAMPLES.
- Choose only from the 5 candidates listed above
- Be concise but clear in your reasoning
- Focus on what the user is actually trying to do or ask"""

        return prompt
    
    async def _call_openai_api(self, prompt: str) -> str:
        """
        Call OpenAI API with the constructed prompt using v1.0+ API.
        
        Args:
            prompt: Formatted prompt string
            
        Returns:
            LLM response text
        """
        try:
            logger.info("🔄 Calling OpenAI API...")
            
            # Make async call to OpenAI using v1.0+ API
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a helpful intent classification assistant. Follow the instructions exactly and respond in the specified format."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            
            response_text = response.choices[0].message.content.strip()
            
            logger.info(f"✅ OpenAI API response received ({len(response_text)} chars)")
            logger.debug(f"Response: {response_text}")
            
            return response_text
            
        except Exception as e:
            logger.error(f"❌ OpenAI API call failed: {str(e)}")
            raise
    
    def _parse_llm_response(self, response_text: str, top_classes: List[Dict]) -> Dict:
        """
        Parse LLM response into structured format.
        
        Args:
            response_text: Raw LLM response
            top_classes: Original candidate classes
            
        Returns:
            Parsed result dictionary
        """
        try:
            # Initialize default values
            intent = top_classes[0]['intent']  # Fallback to top ML prediction
            confidence = 0.7  # Default confidence for LLM decisions
            reasoning = "LLM classification"
            
            # Parse intent
            intent_match = re.search(r'Intent:\s*([A-Z_]+)', response_text)
            if intent_match:
                parsed_intent = intent_match.group(1).strip()
                
                # Verify intent is in our candidates
                valid_intents = [cls['intent'] for cls in top_classes]
                if parsed_intent in valid_intents:
                    intent = parsed_intent
                else:
                    logger.warning(f"⚠️ LLM suggested invalid intent: {parsed_intent}")
                    logger.warning(f"   Using fallback: {intent}")
            
            # Parse confidence level
            confidence_match = re.search(r'Confidence:\s*(High|Medium|Low)', response_text, re.IGNORECASE)
            if confidence_match:
                confidence_level = confidence_match.group(1).lower()
                # Map to numeric confidence
                confidence_mapping = {
                    'high': 0.85,
                    'medium': 0.70, 
                    'low': 0.60
                }
                confidence = confidence_mapping.get(confidence_level, 0.7)
            
            # Parse reasoning
            reasoning_match = re.search(r'Reasoning:\s*(.+)', response_text, re.DOTALL)
            if reasoning_match:
                reasoning = reasoning_match.group(1).strip()
                # Clean up reasoning (remove extra whitespace, limit length)
                reasoning = ' '.join(reasoning.split())
                if len(reasoning) > 200:
                    reasoning = reasoning[:200] + "..."
            
            result = {
                "intent": intent,
                "confidence": confidence,
                "reasoning": reasoning
            }
            
            logger.info(f"✅ Parsed LLM response: {intent} ({confidence:.2f})")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error parsing LLM response: {str(e)}")
            logger.error(f"Raw response: {response_text}")
            
            # Return fallback result
            return {
                "intent": top_classes[0]['intent'],
                "confidence": 0.6,
                "reasoning": f"LLM parsing error, using ML fallback: {str(e)}"
            }
    
    def _validate_intent(self, intent: str, valid_intents: List[str]) -> bool:
        """
        Validate that the intent is in our known set.
        
        Args:
            intent: Intent to validate
            valid_intents: List of valid intent names
            
        Returns:
            True if valid, False otherwise
        """
        return intent in valid_intents
    
    async def health_check(self) -> bool:
        """
        Perform health check on LLM service.
        
        Returns:
            True if LLM is accessible, False otherwise
        """
        try:
            test_prompt = "Test prompt for health check. Respond with: OK"
            
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model,
                messages=[{"role": "user", "content": test_prompt}],
                max_tokens=10,
                temperature=0.0
            )
            
            response_text = response.choices[0].message.content.strip()
            
            logger.info(f"✅ LLM health check passed: {response_text}")
            return True
            
        except Exception as e:
            logger.error(f"❌ LLM health check failed: {str(e)}")
            return False
    
    def get_token_estimate(self, text: str) -> int:
        """
        Estimate token count for text (rough approximation).
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            Estimated token count
        """
        # Rough estimation: 1 token ≈ 4 characters for English text
        return len(text) // 4
    
    def build_debug_info(self, query: str, top_classes: List[Dict]) -> Dict:
        """
        Build debug information for troubleshooting.
        
        Args:
            query: Original query
            top_classes: Top candidate classes
            
        Returns:
            Debug information dictionary
        """
        prompt = self._build_prompt(query, top_classes)
        
        return {
            "query": query,
            "top_classes": top_classes,
            "prompt_length": len(prompt),
            "estimated_tokens": self.get_token_estimate(prompt),
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }


if __name__ == "__main__":
    # Test the LLM handler
    print("🧪 Testing LLM Handler")
    print("=" * 30)
    
    # This would be used for testing during development
    # handler = LLMHandler(api_key="test-key")
    
    print("✅ LLM Handler module loaded successfully!")