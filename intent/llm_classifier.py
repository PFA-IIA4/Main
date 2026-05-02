"""
LLM-Based Intent Classifier for Voice-Controlled Robot System

Uses TinyLlama 1.1B running via llama.cpp for semantic intent classification.
Provides robust error handling and response validation for production use.
"""

import json
import subprocess
import os
import logging
import time
import hashlib
from pathlib import Path
from typing import Dict, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from functools import lru_cache

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    """Structured classification result"""
    intent: str
    confidence: float
    reason: str
    tokens_generated: int = 0
    inference_time_ms: float = 0.0
    model_used: str = "llm"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LLMIntentClassifier:
    """
    Production-grade LLM-based intent classifier.
    
    Features:
    - Subprocess-based llama-cli invocation (isolated, safe)
    - Strict JSON response parsing with validation
    - Inference timeout and error recovery
    - Response caching for identical inputs
    - Comprehensive logging and diagnostics
    """
    
    # Valid intents for validation
    VALID_INTENTS = {
        "START_SESSION",
        "STOP_SESSION", 
        "GET_STATS",
        "BREAK",
        "NAVIGATE",
        "RAG_QUERY",
        "UNKNOWN"
    }

    # JSON schema to force structured output from llama-cli
    JSON_SCHEMA = {
        "type": "object",
        "properties": {
            "intent": {
                "type": "string",
                "enum": sorted(list(VALID_INTENTS)),
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
            },
            "reason": {
                "type": "string",
                "maxLength": 80,
            },
        },
        "required": ["intent", "confidence", "reason"],
        "additionalProperties": False,
    }
    
    # System prompt designed for reliable JSON output
    SYSTEM_PROMPT = """You are an intent classifier for a voice-controlled Raspberry Pi robot.
Your job is to analyze spoken commands and classify them into one of 7 intent categories.
You MUST respond ONLY with valid JSON, no markdown, no explanations before or after.

Available intents:
1. START_SESSION - Commands to begin study/work tracking (e.g., "start session", "begin studying")
2. STOP_SESSION - Commands to end tracking (e.g., "stop session", "I'm done")
3. GET_STATS - Requests for statistics (e.g., "show statistics", "how am I doing")
4. BREAK - Take a break command (e.g., "take a break", "pause")
5. NAVIGATE - Robot movement (e.g., "move forward 3 meters", "turn left 90 degrees")
6. RAG_QUERY - Questions about documents/knowledge (e.g., "what is PID", "explain this")
7. UNKNOWN - Unclear or unsupported commands

Guidance:
- STOP_SESSION when user is done for the day or ending a session ("enough for today", "wrap it up").
- BREAK when the user wants a short rest but not ending the session ("rest now", "pause").
- START_SESSION when the user is beginning work or study ("start studying", "let's get to work").

Examples:
Speech: "start session"
Response: {"intent": "START_SESSION", "confidence": 0.95, "reason": "Begin tracking"}

Speech: "I think I should start studying now"
Response: {"intent": "START_SESSION", "confidence": 0.93, "reason": "Begin studying"}

Speech: "enough for today"
Response: {"intent": "STOP_SESSION", "confidence": 0.92, "reason": "Ending for the day"}

Speech: "wrap it up"
Response: {"intent": "STOP_SESSION", "confidence": 0.90, "reason": "End session"}

Speech: "I should take some rest now"
Response: {"intent": "BREAK", "confidence": 0.90, "reason": "Short rest"}

Speech: "pause for five minutes"
Response: {"intent": "BREAK", "confidence": 0.88, "reason": "Take a break"}

Speech: "show my stats"
Response: {"intent": "GET_STATS", "confidence": 0.90, "reason": "Request stats"}

Speech: "move forward 2 meters and turn left 90 degrees"
Response: {"intent": "NAVIGATE", "confidence": 0.91, "reason": "Move robot"}

Speech: "what is PID control"
Response: {"intent": "RAG_QUERY", "confidence": 0.89, "reason": "Knowledge question"}

Speech: "hey there"
Response: {"intent": "UNKNOWN", "confidence": 0.60, "reason": "No command"}

Your response MUST be exactly this JSON format:
{"intent": "INTENT_NAME", "confidence": 0.95, "reason": "Brief explanation"}

confidence must be 0.0-1.0. reason must be under 50 characters."""

    def __init__(
        self,
        model_path: str,
        llama_cli_path: str,
        max_tokens: int = 150,
        temperature: float = 0.3,
        timeout_seconds: int = 5,
        cache_size: int = 256
    ):
        """
        Initialize LLM classifier.
        
        Args:
            model_path: Path to GGUF model file
            llama_cli_path: Path to llama-cli binary
            max_tokens: Maximum tokens to generate
            temperature: Temperature for sampling (0.0-1.0)
            timeout_seconds: Timeout for inference
            cache_size: LRU cache size for responses
        """
        self.model_path = Path(model_path)
        self.llama_cli_path = Path(llama_cli_path)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self.cache_size = cache_size
        
        # Validation
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        if not self.llama_cli_path.exists():
            raise FileNotFoundError(f"llama-cli not found: {self.llama_cli_path}")
        
        logger.debug("LLMIntentClassifier initialized")
        logger.debug(f"  Model: {self.model_path}")
        logger.debug(f"  Binary: {self.llama_cli_path}")
        logger.debug(f"  Timeout: {self.timeout_seconds}s")
        
        # Initialize cache
        self._classification_cache = {}
        self._stats = {
            "total_calls": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "successful_inferences": 0,
            "failed_inferences": 0,
            "parse_errors": 0,
            "timeouts": 0
        }
    
    def classify(self, text: str) -> ClassificationResult:
        """
        Classify intent from text input.
        
        Args:
            text: User spoken command
            
        Returns:
            ClassificationResult with intent, confidence, and reason
        """
        self._stats["total_calls"] += 1
        
        # Try cache first
        cache_key = self._hash_input(text)
        if cache_key in self._classification_cache:
            self._stats["cache_hits"] += 1
            logger.debug(f"Cache hit for: {text[:50]}")
            return self._classification_cache[cache_key]
        
        self._stats["cache_misses"] += 1
        
        try:
            # Run inference
            start_time = time.time()
            response = self._run_llama_inference(text)
            inference_time = (time.time() - start_time) * 1000
            
            # Parse response
            result = self._parse_llama_response(response, inference_time)
            
            # Validate result
            if not self._validate_result(result):
                logger.debug(f"Invalid result returned: {result}")
                result = ClassificationResult(
                    intent="UNKNOWN",
                    confidence=0.0,
                    reason="LLM returned invalid intent",
                    inference_time_ms=inference_time
                )
            
            self._stats["successful_inferences"] += 1
            
            # Cache and return
            if len(self._classification_cache) < self.cache_size:
                self._classification_cache[cache_key] = result
            
            return result
            
        except subprocess.TimeoutExpired:
            self._stats["timeouts"] += 1
            logger.debug(f"Inference timeout for: {text[:50]}")
            return ClassificationResult(
                intent="UNKNOWN",
                confidence=0.0,
                reason="LLM inference timeout"
            )
        except json.JSONDecodeError as e:
            self._stats["parse_errors"] += 1
            logger.debug(f"JSON parse error: {e}")
            return ClassificationResult(
                intent="UNKNOWN",
                confidence=0.0,
                reason="LLM response parse error"
            )
        except Exception as e:
            self._stats["failed_inferences"] += 1
            logger.debug(f"Inference error: {e}", exc_info=True)
            return ClassificationResult(
                intent="UNKNOWN",
                confidence=0.0,
                reason=f"LLM error: {str(e)[:30]}"
            )
    
    def _run_llama_inference(self, text: str) -> str:
        """
        Run llama-cli subprocess to get LLM response.
        
        Args:
            text: Input text to classify
            
        Returns:
            LLM response string
        """
        # Build prompt with system instructions
        prompt = f"{self.SYSTEM_PROMPT}\n\nSpeech: \"{text}\"\nResponse:"
        
        # Build llama-cli command
        schema_str = json.dumps(self.JSON_SCHEMA, separators=(",", ":"))
        cmd = [
            str(self.llama_cli_path),
            "-m", str(self.model_path),
            "-p", prompt,
            "-n", str(self.max_tokens),
            "-t", "1",  # Single thread for deterministic behavior
            "--temp", str(self.temperature),
            "--no-conversation",
            "--single-turn",
            "--no-display-prompt",
            "--no-show-timings",
            "--log-disable",
            "--simple-io",
            "--json-schema", schema_str,
        ]
        
        logger.debug(f"Running: {' '.join(cmd[:5])}...")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                stdin=subprocess.DEVNULL
            )
            
            if result.returncode != 0:
                error_text = (result.stderr or result.stdout or "").strip()
                raise RuntimeError(f"llama-cli failed: {error_text}")

            stdout = (result.stdout or "").strip()
            stderr = (result.stderr or "").strip()
            combined = "\n".join(part for part in (stdout, stderr) if part).strip()
            if not combined:
                raise RuntimeError("llama-cli returned no output")

            return combined
            
        except subprocess.TimeoutExpired:
            logger.debug(f"llama-cli timeout after {self.timeout_seconds}s")
            raise
        except FileNotFoundError:
            raise RuntimeError(f"llama-cli not found: {self.llama_cli_path}")
    
    def _parse_llama_response(self, response: str, inference_time: float) -> ClassificationResult:
        """
        Parse LLM response and extract JSON.
        
        Args:
            response: Raw stdout from llama-cli
            inference_time: Time taken for inference in ms
            
        Returns:
            Parsed ClassificationResult
        """
        # Extract JSON from response (llama might output multiple lines)
        json_str = self._extract_json(response)
        
        if not json_str:
            raise json.JSONDecodeError("No JSON found in response", response, 0)
        
        # Parse and validate JSON structure
        data = json.loads(json_str)
        
        # Validate required fields
        if not all(k in data for k in ["intent", "confidence", "reason"]):
            raise ValueError(f"Missing required fields in JSON: {data}")
        
        # Validate and sanitize values
        intent = str(data["intent"]).strip()
        confidence = float(data["confidence"])
        reason = str(data["reason"]).strip()[:100]  # Truncate reason
        
        # Validate confidence range
        confidence = max(0.0, min(1.0, confidence))
        
        # Count tokens (rough estimate from response)
        tokens_generated = len(response.split())
        
        return ClassificationResult(
            intent=intent,
            confidence=confidence,
            reason=reason,
            tokens_generated=tokens_generated,
            inference_time_ms=inference_time
        )
    
    def _extract_json(self, text: str) -> Optional[str]:
        """
        Extract JSON object from text response.
        Handles cases where LLM outputs extra text.
        
        Args:
            text: Response text that may contain JSON
            
        Returns:
            JSON string or None
        """
        text = text.strip()
        
        # Find JSON object
        start_idx = text.find('{')
        if start_idx == -1:
            return None
        
        # Find matching closing brace
        brace_count = 0
        for i in range(start_idx, len(text)):
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    return text[start_idx:i+1]
        
        return None
    
    def _validate_result(self, result: ClassificationResult) -> bool:
        """
        Validate that result contains a valid intent.
        
        Args:
            result: Classification result to validate
            
        Returns:
            True if valid, False otherwise
        """
        if result.intent not in self.VALID_INTENTS:
            logger.debug(f"Invalid intent returned: {result.intent}")
            return False
        
        if not (0.0 <= result.confidence <= 1.0):
            logger.debug(f"Invalid confidence: {result.confidence}")
            return False
        
        if not result.reason or len(result.reason) == 0:
            logger.debug("Empty reason")
            return False
        
        return True
    
    @staticmethod
    def _hash_input(text: str) -> str:
        """Hash input for caching"""
        return hashlib.md5(text.lower().encode()).hexdigest()
    
    def clear_cache(self):
        """Clear the response cache"""
        self._classification_cache.clear()
        logger.debug("Classification cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get classifier statistics"""
        return {
            **self._stats,
            "cache_size": len(self._classification_cache),
            "cache_capacity": self.cache_size,
            "cache_hit_rate": (
                self._stats["cache_hits"] / self._stats["total_calls"] 
                if self._stats["total_calls"] > 0 else 0.0
            )
        }
    
    def reset_stats(self):
        """Reset statistics counters"""
        self._stats = {
            "total_calls": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "successful_inferences": 0,
            "failed_inferences": 0,
            "parse_errors": 0,
            "timeouts": 0
        }


# Module-level instance (lazy initialized)
_instance: Optional[LLMIntentClassifier] = None


def get_classifier(
    model_path: Optional[str] = None,
    llama_cli_path: Optional[str] = None
) -> LLMIntentClassifier:
    """
    Get or create singleton classifier instance.
    
    Args:
        model_path: Path to GGUF model (uses env var if not provided)
        llama_cli_path: Path to llama-cli (uses env var if not provided)
        
    Returns:
        LLMIntentClassifier instance
    """
    global _instance
    
    if _instance is not None:
        return _instance
    
    # Use provided paths or environment variables
    model_path = model_path or os.getenv(
        "LLM_MODEL_PATH",
        "./llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    )
    
    llama_cli_path = llama_cli_path or os.getenv(
        "LLAMA_BIN_PATH",
        "./llama.cpp/build/bin/llama-cli"
    )
    
    _instance = LLMIntentClassifier(
        model_path=model_path,
        llama_cli_path=llama_cli_path,
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", "150")),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.3")),
        timeout_seconds=int(os.getenv("LLM_INFERENCE_TIMEOUT", "5"))
    )
    
    return _instance
