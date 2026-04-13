"""
Chatbot fallback for UNKNOWN intents.
Uses Hugging Face cloud inference via API key,
then a lightweight rule-based responder.
"""

import json
import os
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

def _get_cloud_config() -> Optional[dict]:
    """Return Hugging Face cloud config if API key is set."""
    api_key = os.getenv("HUGGINGFACE_API_KEY", "").strip() or os.getenv(
        "CHATBOT_API_KEY", ""
    ).strip()
    if not api_key:
        return None

    timeout_raw = (
        os.getenv("HUGGINGFACE_TIMEOUT_SECONDS", "").strip()
        or os.getenv("CHATBOT_TIMEOUT_SECONDS", "20").strip()
    )
    try:
        timeout_seconds = max(1, int(timeout_raw))
    except ValueError:
        timeout_seconds = 20

    return {
        "api_key": api_key,
        "url": os.getenv(
            "HUGGINGFACE_API_URL",
            "https://router.huggingface.co/v1/chat/completions",
        ).strip(),
        "model": os.getenv(
            "HUGGINGFACE_MODEL",
            os.getenv("CHATBOT_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
        ).strip(),
        "timeout_seconds": timeout_seconds,
    }


def _cloud_response(user_text: str) -> Optional[str]:
    """Call a Hugging Face cloud chat-completions endpoint."""
    config = _get_cloud_config()
    if config is None:
        return None

    payload = {
        "model": config["model"],
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a helpful robot assistant. Keep responses short, clear, "
                    "and command-oriented when possible."
                ),
            },
            {"role": "user", "content": user_text},
        ],
        "temperature": 0.5,
        "max_tokens": 120,
    }

    headers = {
        "Authorization": f"Bearer {config['api_key']}",
        "Content-Type": "application/json",
    }

    request = Request(
        config["url"],
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with urlopen(request, timeout=config["timeout_seconds"]) as response:
            raw = response.read().decode("utf-8")
        parsed = json.loads(raw)

        choices = parsed.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message", {})
            content = message.get("content", "")
            if isinstance(content, str) and content.strip():
                return content.strip()

        # Compatibility with providers that return output_text.
        output_text = parsed.get("output_text", "")
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()

        return None
    except HTTPError as e:
        details = ""
        try:
            details = e.read().decode("utf-8", errors="ignore")
        except Exception:
            pass
        print(f"[CHATBOT] Cloud HTTP error {e.code}: {details}")
        return None
    except URLError as e:
        print(f"[CHATBOT] Cloud network error: {e}")
        return None
    except Exception as e:
        print(f"[CHATBOT] Cloud response error: {e}")
        return None


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
    cloud = _cloud_response(user_text)
    if cloud:
        return cloud

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
