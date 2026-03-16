"""
Main integration module.
STT → Intent Classification → Entity Extraction → Action Dispatcher → Chatbot Fallback
"""

import sys


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

    # Classify intent
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
