"""
Chatbot fallback for UNKNOWN intents.
Uses Hugging Face conversational model.
Future option: Google AI API if free.
"""

from typing import Optional

_pipeline = None


def _load_pipeline():
    """Lazy-load the Hugging Face conversational pipeline."""
    global _pipeline
    if _pipeline is None:
        try:
            from transformers import pipeline

            _pipeline = pipeline(
                "text-generation",
                model="microsoft/DialoGPT-small",
                max_new_tokens=100,
            )
        except Exception as e:
            print(f"[CHATBOT] Failed to load HuggingFace model: {e}")
            _pipeline = "unavailable"


def get_response(user_text: str) -> str:
    """
    Generate a chatbot response for an unrecognized command.

    Parameters
    ----------
    user_text : str
        The user's unrecognized speech text.

    Returns
    -------
    str
        Chatbot response or a helpful fallback message.
    """
    _load_pipeline()

    if _pipeline == "unavailable" or _pipeline is None:
        return _fallback_response(user_text)

    try:
        result = _pipeline(user_text)
        if result and isinstance(result, list) and len(result) > 0:
            generated = result[0].get("generated_text", "").strip()
            # Remove the prompt from the output if echoed back
            if generated.startswith(user_text):
                generated = generated[len(user_text):].strip()
            return generated if generated else _fallback_response(user_text)
        return _fallback_response(user_text)
    except Exception as e:
        print(f"[CHATBOT] Error generating response: {e}")
        return _fallback_response(user_text)


def _fallback_response(user_text: str) -> str:
    """Simple rule-based fallback when ML model is unavailable."""
    text_lower = user_text.lower()
    if any(w in text_lower for w in ["hello", "hi", "hey"]):
        return "Hello! I'm your study robot assistant. Try commands like 'start session' or 'move forward 2 meters'."
    if any(w in text_lower for w in ["help", "what can you do"]):
        return (
            "I can: start/stop study sessions, show stats, take breaks, "
            "and navigate. Try 'start session' or 'move forward 3 meters and turn 90 degrees'."
        )
    if "move" in text_lower or "forward" in text_lower:
        return "Please provide distance and angle. Example: 'move forward 2 meters and turn 90 degrees'."
    return "I didn't understand that. Try commands like 'start session', 'take a break', or 'move forward 5 meters'."


# Optional: Google AI integration placeholder
def get_response_google_ai(user_text: str, api_key: Optional[str] = None) -> str:
    """
    Future Google AI integration.
    Will be activated when a free API option becomes available.
    """
    raise NotImplementedError("Google AI integration is not yet available.")


if __name__ == "__main__":
    test_inputs = [
        "hello",
        "what can you do",
        "move forward",
        "tell me a joke",
    ]
    for text in test_inputs:
        print(f"User: {text}")
        print(f"Bot:  {get_response(text)}")
        print()
