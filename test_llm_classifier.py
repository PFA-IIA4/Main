"""
Integration tests for the LLM-based intent classifier using Hugging Face API.
"""

import pytest
import os
import sys
from unittest.mock import patch, MagicMock
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestLLMClassifier:
    """Test HF API classifier logic using mocks"""
    
    @pytest.fixture
    def mock_requests_post(self):
        with patch('requests.post') as mock_post:
            yield mock_post
            
    @pytest.fixture
    def classifier(self):
        from intent.llm_classifier import get_classifier
        return get_classifier()
        
    def test_classifier_initialization(self, classifier):
        assert classifier is not None
        assert "START_SESSION" in classifier.VALID_INTENTS
        assert "RESUME_SESSION" in classifier.VALID_INTENTS
        
    def test_classify_start_session(self, classifier, mock_requests_post):
        # Setup mock network response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": '{"intent": "START_SESSION", "confidence": 0.95, "reason": "Begin tracking"}'
                }
            }]
        }
        mock_requests_post.return_value = mock_response
        
        result = classifier.classify("start session")
        assert result.intent == "START_SESSION"
        assert result.confidence == 0.95
        assert result.parameters == {}
        
    def test_classify_navigate(self, classifier, mock_requests_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": '{"intent": "NAVIGATE", "confidence": 0.99, "reason": "Move", "parameters": {"distance": 5, "angle": 0}}'
                }
            }]
        }
        mock_requests_post.return_value = mock_response
        
        result = classifier.classify("move forward 5 meters")
        assert result.intent == "NAVIGATE"
        assert result.parameters["distance"] == 5
        
    def test_classify_chatbot_fallback(self, classifier, mock_requests_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": '{"intent": "CHATBOT", "confidence": 0.99, "reason": "Small talk", "response": "Hello! How can I help?"}'
                }
            }]
        }
        mock_requests_post.return_value = mock_response
        
        result = classifier.classify("hi there")
        assert result.intent == "CHATBOT"
        assert result.response == "Hello! How can I help?"
        
    def test_network_failure(self, classifier, mock_requests_post):
        import requests
        mock_requests_post.side_effect = requests.exceptions.ConnectionError("Connection Failed")
        
        result = classifier.classify("ping")
        assert result.intent == "UNKNOWN"
        assert result.reason == "API Network Error"

class TestAppPipeline:
    def test_predict_wrapper(self):
        """Test that the intent_classifier.py wrapper handles HF parameters format"""
        from intent.intent_classifier import IntentClassifier
        with patch('intent.llm_classifier.LLMIntentClassifier.classify') as mock_classify:
            from intent.llm_classifier import ClassificationResult
            mock_classify.return_value = ClassificationResult(
                intent="NAVIGATE", 
                confidence=0.9, 
                parameters={"distance": 10}
            )
            
            wrapper = IntentClassifier()
            result_dict = wrapper.predict("move 10 meters")
            
            assert isinstance(result_dict, dict)
            assert result_dict["intent"] == "NAVIGATE"
            assert result_dict["parameters"]["distance"] == 10

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
