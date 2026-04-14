"""
Main integration module.
STT → Intent Classification → Entity Extraction → Action Dispatcher → Chatbot Fallback
"""

import sys


_RAG_TRIGGER_PHRASES = [
    "what is",
    "explain",
    "define",
    "summarize",
    "summary",
    "what does",
    "tell me about",
    "give details",
    "what is written",
]

_RAG_CONTEXT_HINTS = [
    "course",
    "chapter",
    "lesson",
    "document",
    "pdf",
    "file",
    "files",
    "notes",
    "from my course",
    "from my notes",
    "uploaded",
    "content",
    "page",
]

_SESSION_COMMAND_PHRASES = [
    "start session",
    "begin session",
    "start studying",
    "begin studying",
    "start study mode",
    "start focus mode",
    "stop session",
    "end session",
    "stop studying",
    "end study mode",
    "finish session",
    "terminate session",
    "stop the study timer",
]

_STATS_COMMAND_PHRASES = [
    "get stats",
    "show statistics",
    "display stats",
    "how am i doing",
    "show my progress",
    "what are my stats",
    "session statistics",
    "progress report",
    "study report",
    "performance stats",
    "summary of sessions",
    "give me a summary",
]

_BREAK_COMMAND_PHRASES = [
    "take a break",
    "i need a break",
    "break time",
    "pause session",
    "pause the session",
    "pause studying",
    "break now",
    "rest time",
    "quick break",
]

_NAVIGATION_VERBS = [
    "move",
    "go",
    "turn",
    "rotate",
    "drive",
    "travel",
    "advance",
    "proceed",
    "navigate",
    "head",
]

_NAVIGATION_HINTS = [
    "meter",
    "meters",
    "m ",
    "degree",
    "degrees",
    "left",
    "right",
    "forward",
    "ahead",
    "straight",
    "north",
    "south",
    "east",
    "west",
]


def _contains_any(text: str, phrases) -> bool:
    return any(phrase in text for phrase in phrases)


def _looks_like_navigation_command(text: str) -> bool:
    return _contains_any(text, _NAVIGATION_VERBS) and _contains_any(
        text, _NAVIGATION_HINTS
    )


def _is_reserved_robot_command(text: str) -> bool:
    """Return True for known non-RAG robot commands.

    This guard protects existing session/stats/break/navigation behaviors
    from being overridden by RAG pre-routing.
    """
    if _contains_any(text, _SESSION_COMMAND_PHRASES):
        return True
    if _contains_any(text, _STATS_COMMAND_PHRASES):
        return True
    if _contains_any(text, _BREAK_COMMAND_PHRASES):
        return True
    return _looks_like_navigation_command(text)


def is_rag_query(text: str) -> bool:
    """Rule-based detector for document QA requests (RAG intent)."""
    text = text.lower().strip()
    if not text:
        return False

    if _is_reserved_robot_command(text):
        return False

    has_rag_trigger = _contains_any(text, _RAG_TRIGGER_PHRASES)
    has_context_hint = _contains_any(text, _RAG_CONTEXT_HINTS)
    return has_rag_trigger or has_context_hint


def process_text(text: str, intent_classifier, verbose: bool = True) -> str:
    """
    Process recognized text through the full pipeline.

    Parameters
    ----------
    text : str
        Recognized speech text.
    intent_classifier : IntentClassifier
        Loaded intent classifier instance.
    verbose : bool
        Print detailed console output.

    Returns
    -------
    str
        Action result message.
    """
    from entity.entity_extractor import extract_entities, has_required_entities
    from action.dispatcher import dispatch
    from chatbot.chatbot_handler import get_response

    # Rule-based pre-routing for document QA before classifier.
    if is_rag_query(text):
        intent = "RAG_QUERY"
        confidence = 1.0
    else:
        result = intent_classifier.predict(text)
        intent = result["intent"]
        confidence = result["confidence"]

    entities = None

    # Extract entities for NAVIGATE intent
    if intent == "NAVIGATE":
        entities = extract_entities(text)
        if not has_required_entities(entities):
            intent = "UNKNOWN"

    if verbose:
        print(f"\nYou said: {text}")
        print(f"Intent: {intent}")
        print(f"Confidence: {confidence}")
        if entities and intent == "NAVIGATE":
            print(f"Distance: {entities.get('distance')}")
            print(f"Angle: {entities.get('angle')}")

    # Dispatch action
    action_result = dispatch(intent, entities=entities, text=text)

    # Chatbot fallback for UNKNOWN
    if action_result == "CHATBOT_FALLBACK":
        chatbot_response = get_response(text)
        if verbose:
            print(f"Action: Chatbot invoked")
            print(f"Response: {chatbot_response}")
        return chatbot_response

    if verbose:
        print(f"Action: {action_result}")

    return action_result


def run_with_stt():
    """Run the full pipeline with live microphone input."""
    from stt.vosk_stt import create_recognizer, listen
    from intent.intent_classifier import IntentClassifier

    print("=" * 50)
    print("  Voice-Controlled Robotic System")
    print("  Say 'stop session' or press Ctrl+C to exit")
    print("=" * 50)

    clf = IntentClassifier()
    recognizer = create_recognizer()

    def on_partial(text):
        sys.stdout.write(f"\r  (listening) {text}   ")
        sys.stdout.flush()

    def on_result(text):
        sys.stdout.write("\r" + " " * 60 + "\r")  # clear partial line
        process_text(text, clf)
        return None  # keep listening

    try:
        listen(recognizer, on_partial=on_partial, on_result=on_result)
    except KeyboardInterrupt:
        print("\n[MAIN] Stopped by user.")


def run_text_mode():
    """Run the pipeline with typed text input (no microphone)."""
    from intent.intent_classifier import IntentClassifier

    print("=" * 50)
    print("  Voice-Controlled Robotic System (Text Mode)")
    print("  Type 'quit' to exit")
    print("=" * 50)

    clf = IntentClassifier()

    while True:
        try:
            text = input("\nYou: ").strip()
            if not text:
                continue
            if text.lower() in ("quit", "exit"):
                print("Goodbye!")
                break
            process_text(text, clf)
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    if "--text" in sys.argv:
        run_text_mode()
    else:
        try:
            run_with_stt()
        except (FileNotFoundError, ImportError) as e:
            print(f"[MAIN] STT unavailable: {e}")
            print("[MAIN] Falling back to text mode…")
            run_text_mode()
