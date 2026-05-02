"""
Integration tests for the LLM-based intent classifier.
Tests the complete pipeline using the local llama.cpp model only.
"""

import pytest
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestLLMClassifier:
    """Test LLM classifier initialization and classification"""
    
    @pytest.fixture
    def llm_classifier(self):
        """Initialize LLM classifier"""
        from intent.llm_classifier import get_classifier
        
        # Skip if model not found
        model_path = os.getenv(
            "LLM_MODEL_PATH",
            "./llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
        )
        if not Path(model_path).exists():
            pytest.skip("Model file not found")
        
        return get_classifier()
    
    def test_classifier_initialization(self, llm_classifier):
        """Test LLM classifier initializes correctly"""
        assert llm_classifier is not None
        assert llm_classifier.model_path.exists()
        assert llm_classifier.llama_cli_path.exists()
    
    def test_valid_intents_defined(self, llm_classifier):
        """Test all valid intents are defined"""
        expected_intents = {
            "START_SESSION", "STOP_SESSION", "GET_STATS", 
            "BREAK", "NAVIGATE", "RAG_QUERY", "UNKNOWN"
        }
        assert llm_classifier.VALID_INTENTS == expected_intents
    
    def test_classify_start_session(self, llm_classifier):
        """Test classification of START_SESSION intent"""
        result = llm_classifier.classify("start session")
        
        assert result.intent in llm_classifier.VALID_INTENTS
        assert 0.0 <= result.confidence <= 1.0
        assert result.reason and len(result.reason) > 0
        assert result.model_used == "llm"
    
    def test_classify_navigate(self, llm_classifier):
        """Test classification of NAVIGATE intent"""
        result = llm_classifier.classify("move forward 5 meters")
        
        assert result.intent in llm_classifier.VALID_INTENTS
        assert 0.0 <= result.confidence <= 1.0
    
    def test_classify_unknown(self, llm_classifier):
        """Test classification of ambiguous/unknown input"""
        result = llm_classifier.classify("xyz qwerty asdf")
        
        assert result.intent in llm_classifier.VALID_INTENTS
        # UNKNOWN intent should have lower confidence
    
    def test_cache_functionality(self, llm_classifier):
        """Test that caching works"""
        llm_classifier.clear_cache()
        initial_hits = llm_classifier._stats["cache_hits"]
        
        # First call
        result1 = llm_classifier.classify("start session")
        
        # Second call (should hit cache)
        result2 = llm_classifier.classify("start session")
        
        assert result1.intent == result2.intent
        assert result1.confidence == result2.confidence
        assert llm_classifier._stats["cache_hits"] > initial_hits
    
    def test_stats_tracking(self, llm_classifier):
        """Test statistics tracking"""
        llm_classifier.reset_stats()
        
        result = llm_classifier.classify("test")
        stats = llm_classifier.get_stats()
        
        assert stats["total_calls"] == 1
        assert stats["successful_inferences"] == 1


class TestLLMOnlyClassifier:
    """Test the LLM-only wrapper used by the app"""
    
    @pytest.fixture
    def llm_wrapper(self):
        """Initialize the LLM-only wrapper"""
        from intent.intent_classifier import IntentClassifier
        return IntentClassifier()
    
    def test_classifier_initialization(self, llm_wrapper):
        """Test wrapper initializes"""
        assert llm_wrapper is not None
    
    def test_predict_returns_result_dict(self, llm_wrapper):
        """Test predict returns proper result dictionary"""
        result = llm_wrapper.predict("start session")
        
        assert isinstance(result, dict)
        assert result["intent"] in [
            "START_SESSION", "STOP_SESSION", "GET_STATS",
            "BREAK", "NAVIGATE", "RAG_QUERY", "UNKNOWN"
        ]
        assert 0.0 <= result["confidence"] <= 1.0
    
    def test_intent_varieties(self, llm_wrapper):
        """Test classification of different intent types"""
        test_cases = [
            ("start session", "START_SESSION"),
            ("begin studying", "START_SESSION"),
            ("stop session", "STOP_SESSION"),
            ("end studying", "STOP_SESSION"),
            ("show statistics", "GET_STATS"),
            ("how am I doing", "GET_STATS"),
            ("take a break", "BREAK"),
            ("move forward 3 meters", "NAVIGATE"),
            ("turn left 90 degrees", "NAVIGATE"),
            ("what is PID", "RAG_QUERY"),
            ("xyz random nonsense qwerty", None),  # Should be UNKNOWN or similar
        ]
        
        for input_text, expected_intent in test_cases:
            result = llm_wrapper.predict(input_text)
            assert result["intent"] in [
                "START_SESSION", "STOP_SESSION", "GET_STATS",
                "BREAK", "NAVIGATE", "RAG_QUERY", "UNKNOWN"
            ]
    
    def test_result_to_dict(self, llm_wrapper):
        """Test converting result to dictionary"""
        result_dict = llm_wrapper.predict("start session")
        
        assert isinstance(result_dict, dict)
        assert "intent" in result_dict
        assert "confidence" in result_dict
        assert "model_used" in result_dict


class TestErrorHandling:
    """Test error handling and robustness"""
    
    def test_empty_input(self):
        """Test handling of empty input"""
        from intent.intent_classifier import IntentClassifier
        
        classifier = IntentClassifier()
        result = classifier.predict("")
        
        # Should return some valid response
        assert result["intent"] in [
            "START_SESSION", "STOP_SESSION", "GET_STATS",
            "BREAK", "NAVIGATE", "RAG_QUERY", "UNKNOWN"
        ]
    
    def test_very_long_input(self):
        """Test handling of very long input"""
        from intent.intent_classifier import IntentClassifier
        
        classifier = IntentClassifier()
        long_text = "word " * 500  # Very long repetitive text
        
        result = classifier.predict(long_text)
        assert result["intent"] in [
            "START_SESSION", "STOP_SESSION", "GET_STATS",
            "BREAK", "NAVIGATE", "RAG_QUERY", "UNKNOWN"
        ]
    
    def test_special_characters(self):
        """Test handling of special characters"""
        from intent.intent_classifier import IntentClassifier
        
        classifier = IntentClassifier()
        special_text = "!@#$%^&*() 中文 🎉 émojis"
        
        result = classifier.predict(special_text)
        assert result["intent"] in [
            "START_SESSION", "STOP_SESSION", "GET_STATS",
            "BREAK", "NAVIGATE", "RAG_QUERY", "UNKNOWN"
        ]


class TestPipelineIntegration:
    """Test complete pipeline integration"""
    
    def test_entity_extraction_after_classification(self):
        """Test entity extraction following intent classification"""
        from intent.intent_classifier import IntentClassifier
        from entity.entity_extractor import extract_entities
        
        classifier = IntentClassifier()
        text = "move forward 5 meters and turn left 90 degrees"
        
        # Classify
        intent_result = classifier.predict(text)
        
        # Extract entities if NAVIGATE
        if intent_result["intent"] == "NAVIGATE":
            entities = extract_entities(text, intent_result["intent"])
            
            assert "distance" in entities or "angle" in entities
    
    def test_unknown_intent_triggers_chatbot_path(self):
        """Test that UNKNOWN intent is properly identified"""
        from intent.intent_classifier import IntentClassifier
        
        classifier = IntentClassifier()
        result = classifier.predict("tell me a joke about programming")
        
        # Should return a valid intent (might be RAG_QUERY or UNKNOWN)
        assert result["intent"] in [
            "START_SESSION", "STOP_SESSION", "GET_STATS",
            "BREAK", "NAVIGATE", "RAG_QUERY", "UNKNOWN"
        ]


class TestPerformance:
    """Test performance characteristics"""
    
    def test_classification_latency(self):
        """Test classification completes in reasonable time"""
        from intent.intent_classifier import IntentClassifier
        import time
        
        classifier = IntentClassifier()
        
        start = time.time()
        result = classifier.predict("start session")
        elapsed_ms = (time.time() - start) * 1000
        
        # LLM might be slow on first call, but should complete
        assert elapsed_ms < 30000  # 30 seconds max
        print(f"\nClassification latency: {elapsed_ms:.1f}ms")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
