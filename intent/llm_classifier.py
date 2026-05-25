"""
LLM-Based Intent Classifier for Voice-Controlled Robot System
Uses Hugging Face Inference API for semantic intent classification and entity extraction.
"""

import json
import os
import logging
import time
import requests
from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class ClassificationResult:
    """Structured classification result"""
    intent: str
    confidence: float
    reason: str = ""
    parameters: Dict[str, Any] = None
    response: str = ""
    inference_time_ms: float = 0.0
    model_used: str = "hf_api"
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class LLMIntentClassifier:
    """
    Production-grade LLM-based intent classifier using Hugging Face API.
    """
    
    VALID_INTENTS = {
        "START_SESSION",
        "STOP_SESSION", 
        "RESUME_SESSION",
        "GET_STATS",
        "BREAK",
        "NAVIGATE",
        "RAG_QUERY",
        "CHATBOT",
        "UNKNOWN"
    }

    SYSTEM_PROMPT = """You are the brain of a voice-controlled Raspberry Pi robot.
Your name is Deskmate. You are a helpful, conversational AI robotic assistant.
The user will say a command or ask a question.
Your job is to classify the intent and extract any necessary parameters.
You MUST respond ONLY with valid JSON. No markdown formatting, no explanations outside the JSON object.

Available intents:
1. START_SESSION - Commands to begin study/work tracking ("start session")
2. STOP_SESSION - Commands to end tracking ("stop session", "I'm done")
3. RESUME_SESSION - Resume work after a break ("resume session", "back to work")
4. GET_STATS - Requests for statistics ("how am I doing")
5. BREAK - Take a break ("take a little break")
6. NAVIGATE - Robot movement ("move forward 3 meters", "turn left")
7. RAG_QUERY - Questions about uploaded documents or course content ("what is PID control", "summarize chapter 3")
8. CHATBOT - General conversational talk, greetings, or off-topic questions ("hello", "how are you building your navigation stack")
9. UNKNOWN - Unintelligible noise.

Required Output JSON Format:
{"intent": "INTENT_NAME", "confidence": 0.95, "reason": "Brief explanation", "parameters": {}, "response": ""}

Specific Rules:
- If intent is NAVIGATE, include "distance" (number) and "angle" (number) in parameters if present. Example: {"distance": 3, "angle": -90} for moving forward 3 meters and turning right 90 degrees.
- If intent is CHATBOT, include what you want to say back to the user in the "response" field. Do not use parameters.
- For all other known intents, parameters should be empty {} and response should be "".
- The confidence field should be a float between 0.0 and 1.0.

Examples:
Speech: "start session"
Response: {"intent": "START_SESSION", "confidence": 0.95, "reason": "Begin tracking", "parameters": {}, "response": ""}

Speech: "move forward 2 meters and turn right 90 degrees"
Response: {"intent": "NAVIGATE", "confidence": 0.91, "reason": "Move robot", "parameters": {"distance": 2, "angle": -90}, "response": ""}

Speech: "how are you?"
Response: {"intent": "CHATBOT", "confidence": 0.99, "reason": "Small talk", "parameters": {}, "response": "I'm doing well, ready to help! What's next?"}

Your response MUST be exactly the JSON object, nothing else."""

    def __init__(self):
        self.api_url = os.environ.get("HUGGINGFACE_API_URL", "https://router.huggingface.co/v1/chat/completions")
        self.api_key = os.environ.get("HUGGINGFACE_API_KEY")
        default_models = [
            "Qwen/Qwen2.5-7B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.2",
            "HuggingFaceH4/zephyr-7b-beta",
        ]
        env_model = os.environ.get("HUGGINGFACE_MODEL")
        if env_model:
            self.models = [env_model] + [model for model in default_models if model != env_model]
        else:
            self.models = default_models
        self.timeout_seconds = int(os.environ.get("HUGGINGFACE_TIMEOUT_SECONDS", "10"))
        
        self._stats = {
            "total_calls": 0,
            "successful_inferences": 0,
            "failed_inferences": 0,
            "parse_errors": 0,
            "timeouts": 0
        }
        
    def classify(self, text: str) -> ClassificationResult:
        self._stats["total_calls"] += 1
        
        last_error_reason = "Unknown error"
        last_model = self.models[-1]
        last_inference_time = 0.0
        
        for model in self.models:
            try:
                start_time = time.time()
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": text}
                    ],
                    "response_format": {"type": "json_object"},
                    "max_tokens": 150
                }
                
                response = requests.post(self.api_url, headers=headers, json=payload, timeout=self.timeout_seconds)
                response.raise_for_status()
                inference_time = (time.time() - start_time) * 1000
                last_inference_time = inference_time
                
                data = response.json()
                raw_content = data['choices'][0]['message']['content']
                
                result = self._parse_json(raw_content, inference_time)
                result.model_used = model
                
                if not self._validate_result(result):
                    last_error_reason = "API returned invalid intent"
                    logger.error(f"Invalid intent from model {model}")
                    continue
                
                self._stats["successful_inferences"] += 1
                return result
                
            except requests.exceptions.Timeout:
                self._stats["timeouts"] += 1
                last_error_reason = "API timeout"
                logger.error(f"Hugging Face API Timeout (model={model})")
            except requests.exceptions.RequestException as e:
                last_error_reason = "API Network Error"
                logger.error(f"Hugging Face API Error (model={model}): {e}")
            except json.JSONDecodeError as e:
                self._stats["parse_errors"] += 1
                last_error_reason = "JSON parse error"
                logger.error(f"JSON Parse Error (model={model}): {e}")
            except Exception as e:
                last_error_reason = f"Error: {str(e)[:30]}"
                logger.error(f"Unexpected Error (model={model}): {e}")
        
        self._stats["failed_inferences"] += 1
        return ClassificationResult(
            intent="UNKNOWN",
            confidence=0.0,
            reason=last_error_reason,
            inference_time_ms=last_inference_time,
            model_used=last_model
        )

    def _parse_json(self, response_text: str, inference_time: float) -> ClassificationResult:
        # Some models markdown-format their JSON
        text = response_text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
            
        data = json.loads(text.strip())
        
        intent = str(data.get("intent", "UNKNOWN")).strip()
        confidence = float(data.get("confidence", 1.0))
        reason = str(data.get("reason", "")).strip()[:100]
        parameters = data.get("parameters", {})
        response = data.get("response", "")
        
        confidence = max(0.0, min(1.0, confidence))
        
        if not isinstance(parameters, dict):
            parameters = {}
            
        return ClassificationResult(
            intent=intent,
            confidence=confidence,
            reason=reason,
            parameters=parameters,
            response=response,
            inference_time_ms=inference_time
        )

    def _validate_result(self, result: ClassificationResult) -> bool:
        if result.intent not in self.VALID_INTENTS:
            return False
        return True

_instance: Optional[LLMIntentClassifier] = None

def get_classifier() -> LLMIntentClassifier:
    global _instance
    if _instance is None:
        _instance = LLMIntentClassifier()
    return _instance
