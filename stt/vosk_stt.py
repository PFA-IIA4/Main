"""
Live microphone speech-to-text using Vosk 0.22.
Continuous streaming with partial and final transcription.
"""

import json
import os
import queue
import sys
import re
import time

import sounddevice as sd
from vosk import Model, KaldiRecognizer
import speech_recognition as sr

# Choose which model to use here:
# "vosk_model" is the extracted folder path for the model
SELECTED_MODEL = "vosk_model"

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
    """Flexible fuzzy matching for 'hey deskmate' or simple greetings."""
    text = text.lower()
    # Accept simple greetings like 'hello', 'hi', or 'hey' on their own
    if re.search(r'^\s*(hello|hi|hey)\s*$', text):
        return True
    # Check for presence of hey and something sounding like deskmate
    if re.search(r'\b(hey|hi|hello|ok|okay)\b.*\b(desk|disk|test|this|just|guess)\s*(mate|make|made|man|may|me)*\b', text) or \
       re.search(r'\b(hey|hi|hello|ok|okay)\b.*\b(deskmate|this mate|test mate|just made|just make)\b', text):
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
    cooldown_until = 0
    skip_wake_word_next = False

    print("[STT-Hybrid] Listening for wake words 'hey deskmate', 'hello', or 'hi' using Vosk… (Ctrl+C to stop)")
    while True:
        with sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=8000,
            dtype="int16",
            channels=1,
            callback=_audio_callback,
        ):
            detected_wake_word = False
            while True:
                if not skip_wake_word_next:
                    data = audio_queue.get()
                    
                    # Prevent robot's own TTS outputs from triggering the wake word
                    if time.time() < cooldown_until:
                        continue

                    #accept waveform returns True when it has a final result (silence meaning user finished speaking), False for partials
                    if recognizer.AcceptWaveform(data):
                        result = json.loads(recognizer.Result())
                        text = result.get("text", "").strip()
                        
                        if text and is_wake_word(text):
                            detected_wake_word = True
                            if on_partial:
                                on_partial(f"[Vosk Match] {text}" + " " * 20)
                            break
                        elif text:
                            # It was a final phrase, but not the wake word.
                            if on_partial:
                                on_partial(f"[Ignored] {text}" + " " * 20)
                    else:
                        partial = json.loads(recognizer.PartialResult())
                        partial_text = partial.get("partial", "").strip()
                        
                        if on_partial:
                            if partial_text:
                                on_partial(f"[Vosk Partial] {partial_text}" + " " * 20)
                            else:
                                on_partial("\r  (listening) [Waiting...]                    ")
                else:
                    break

        if detected_wake_word or skip_wake_word_next:
            # Flush the queue to discard old audio
            while not audio_queue.empty():
                audio_queue.get()

            if detected_wake_word:
                # Trigger beep (audible to user)
                print("\a\n[STT-Hybrid] Wake word detected! Triggering TTS...")
                sys.stdout.flush()
                # Trigger TTS
                speak("Hello, how can I assist you today?")
            else:
                # We are in conversational bypass mode
                print("\a\n[STT-Hybrid] Continuing conversation... listening directly.")

            # Reset bypass flag
            skip_wake_word_next = False

            # Now switch to Google Web Speech API to get the command
            print("[STT-Hybrid] Activating Google Web Speech API... Listening for actual command...")
            with sr.Microphone() as source:
                # Optional: google_recognizer.adjust_for_ambient_noise(source)
                try:
                    audio = google_recognizer.listen(source, timeout=5, phrase_time_limit=20)
                    print("[STT-Hybrid] Processing with Google...")
                    command_text = google_recognizer.recognize_google(audio)
                    print(f"[STT-Hybrid] Google recognized: {command_text}")
                    if on_result:
                        res = on_result(command_text)
                        if res is False:
                            return
                        elif res == "SKIP_WAKE_WORD":
                            skip_wake_word_next = True
                except sr.WaitTimeoutError:
                    print("[STT-Hybrid] Timeout: No command heard after wake word.")
                except sr.UnknownValueError:
                    print("[STT-Hybrid] Google Speech Recognition could not understand audio.")
                except sr.RequestError as e:
                    print(f"[STT-Hybrid] Could not request results from Google Speech Recognition service; {e}")
            
            print("[STT-Hybrid] Resuming Vosk wake word listening...\n")
            
            # Small pause to allow physical speaker echo to dissipate
            time.sleep(0.5)

            # Reset the KaldiRecognizer to clear out the old partials before continuing
            while not audio_queue.empty():
                audio_queue.get()
            
            # Set a 2 second cooldown window where we deliberately ignore Vosk 
            # so that any lingering echoes from the TTS do not trigger another wake word
            cooldown_until = time.time() + 2.0


if __name__ == "__main__":
    rec = create_recognizer()
    listen(
        rec,
        on_partial=lambda t: print(f"  (partial) {t}"),
        on_result=lambda t: print(f"  >> {t}"),
    )
