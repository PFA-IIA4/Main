"""
Live microphone speech-to-text using Vosk 0.22.
Continuous streaming with partial and final transcription.
"""

import json
import os
import queue
import sys

import sounddevice as sd
from vosk import Model, KaldiRecognizer

SAMPLE_RATE = 16000
MODEL_PATH = os.path.join(os.path.dirname(__file__), "vosk_model")

audio_queue: queue.Queue = queue.Queue()


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
    model = Model(model_path)
    return KaldiRecognizer(model, SAMPLE_RATE)


def listen(recognizer: KaldiRecognizer, on_partial=None, on_result=None):
    """
    Start continuous microphone streaming.

    Parameters
    ----------
    recognizer : KaldiRecognizer
        The Vosk recognizer instance.
    on_partial : callable, optional
        Called with partial transcription text.
    on_result : callable, optional
        Called with final transcription text. Return False to stop listening.
    """
    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=8000,
        dtype="int16",
        channels=1,
        callback=_audio_callback,
    ):
        print("[STT] Listening… (Ctrl+C to stop)")
        while True:
            data = audio_queue.get()
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "").strip()
                if text and on_result:
                    if on_result(text) is False:
                        break
            else:
                partial = json.loads(recognizer.PartialResult())
                partial_text = partial.get("partial", "").strip()
                if partial_text and on_partial:
                    on_partial(partial_text)


if __name__ == "__main__":
    rec = create_recognizer()
    listen(
        rec,
        on_partial=lambda t: print(f"  (partial) {t}"),
        on_result=lambda t: print(f"  >> {t}"),
    )
