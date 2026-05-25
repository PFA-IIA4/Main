"""
Live microphone speech-to-text using Vosk 0.22.
Continuous streaming with partial and final transcription.
"""

import json
import os
import queue
import sys
import re

import sounddevice as sd
from vosk import Model, KaldiRecognizer
import speech_recognition as sr

# Choose which model to use here:
# "vosk-model-small-en-us-0.15" (fast, less accurate)
# "vosk-model-en-us-0.22" (slower, more accurate)
SELECTED_MODEL = "vosk-model-small-en-us-0.15"

SAMPLE_RATE = 16000
MODEL_PATH = os.getenv(
    "VOSK_MODEL_PATH", os.path.join(os.path.dirname(__file__), SELECTED_MODEL)
)

audio_queue: queue.Queue = queue.Queue()


def _is_valid_vosk_model_dir(model_path: str) -> bool:
    """Return True when directory looks like a Vosk model root."""
    required_subdirs = ("am", "conf", "graph")
    return all(os.path.isdir(os.path.join(model_path, name)) for name in required_subdirs)


def _audio_callback(indata, frames, time_info, status):
    """Callback for sounddevice to enqueue raw audio data."""
    if status:
        print(f"[STT] Audio status: {status}", file=sys.stderr)
    audio_queue.put(bytes(indata))


def create_recognizer(model_path: str = MODEL_PATH) -> KaldiRecognizer:
    """Load the Vosk model and return a KaldiRecognizer."""
    if not os.path.isdir(model_path):
        raise FileNotFoundError(
            f"Vosk model not found at '{model_path}'. "
            "Download a model from https://alphacephei.com/vosk/models "
            "and extract it into the vosk_model/ directory."
        )

    if not _is_valid_vosk_model_dir(model_path):
        visible_entries = [entry for entry in os.listdir(model_path) if not entry.startswith(".")]
        found = ", ".join(sorted(visible_entries)[:8]) if visible_entries else "(empty directory)"
        raise FileNotFoundError(
            f"Invalid Vosk model layout at '{model_path}'. "
            "Expected subfolders: am/, conf/, graph/. "
            f"Found: {found}. "
            "If you extracted a zip, move the model contents (not just the parent folder) into this path."
        )

    try:
        model = Model(model_path)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load Vosk model from '{model_path}'. "
            "The model may be incomplete/corrupted or built for a different Vosk version."
        ) from exc

    return KaldiRecognizer(model, SAMPLE_RATE)


def is_wake_word(text: str) -> bool:
    """Flexible fuzzy matching for 'hey deskmate'."""
    text = text.lower()
    # Check for presence of hey and something sounding like deskmate
    if re.search(r'\b(hey|hi|hello)\b.*\b(desk|disk|test|this)\s*(mate|make|made|man)*\b', text) or \
       re.search(r'\b(hey|hi|hello)\b.*\b(deskmate|this mate|test mate)\b', text):
        return True
    return False

def listen(recognizer: KaldiRecognizer, on_partial=None, on_result=None):
    """
    Hybrid continuous microphone streaming.
    Uses Vosk for wake word 'hey deskmate' and Google Web Speech API for actual command.

    Parameters
    ----------
    recognizer : KaldiRecognizer
        The Vosk recognizer instance (used for Wake Word).
    on_partial : callable, optional
        Called with partial transcription text.
    on_result : callable, optional
        Called with final transcription text. Return False to stop listening.
    """
    # Import speak here to avoid circular imports just in case
    from tts.engine import speak

    google_recognizer = sr.Recognizer()

    print("[STT-Hybrid] Listening for wake word 'hey deskmate' using Vosk… (Ctrl+C to stop)")
    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=8000,
        dtype="int16",
        channels=1,
        callback=_audio_callback,
    ):
        while True:
            data = audio_queue.get()
            detected_wake_word = False

            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "").strip()
                if text and on_partial: 
                    # Clear out line buffer if we print partials
                    on_partial(f"[Vosk] {text}")
                if text and is_wake_word(text):
                    detected_wake_word = True
            else:
                partial = json.loads(recognizer.PartialResult())
                partial_text = partial.get("partial", "").strip()
                if partial_text and on_partial:
                    on_partial(f"[Vosk Partial] {partial_text}")
                if partial_text and is_wake_word(partial_text):
                    detected_wake_word = True

            if detected_wake_word:
                # Flush the queue to discard old audio
                while not audio_queue.empty():
                    audio_queue.get()

                # Trigger beep (audible to user)
                print("\a\n[STT-Hybrid] Wake word detected! Triggering TTS...")
                sys.stdout.flush()

                # Trigger TTS
                speak("Hello, how can I assist you today?")

                # Now switch to Google Web Speech API to get the command
                print("[STT-Hybrid] Activating Google Web Speech API... Listening for actual command...")
                with sr.Microphone() as source:
                    # Optional: google_recognizer.adjust_for_ambient_noise(source)
                    try:
                        audio = google_recognizer.listen(source, timeout=5, phrase_time_limit=10)
                        print("[STT-Hybrid] Processing with Google...")
                        command_text = google_recognizer.recognize_google(audio)
                        print(f"[STT-Hybrid] Google recognized: {command_text}")
                        if on_result:
                            if on_result(command_text) is False:
                                break
                    except sr.WaitTimeoutError:
                        print("[STT-Hybrid] Timeout: No command heard after wake word.")
                    except sr.UnknownValueError:
                        print("[STT-Hybrid] Google Speech Recognition could not understand audio.")
                    except sr.RequestError as e:
                        print(f"[STT-Hybrid] Could not request results from Google Speech Recognition service; {e}")
                
                print("[STT-Hybrid] Resuming Vosk wake word listening...\n")
                
                # Reset the KaldiRecognizer to clear out the old partials before continuing
                while not audio_queue.empty():
                    audio_queue.get()


if __name__ == "__main__":
    rec = create_recognizer()
    listen(
        rec,
        on_partial=lambda t: print(f"  (partial) {t}"),
        on_result=lambda t: print(f"  >> {t}"),
    )
