import numpy as np
import wave

# Parameters
fs = 44100  # Sample rate
duration = 1.0  # seconds
frequency = 440.0  # Hz (A4 note)

t = np.linspace(0, duration, int(fs * duration), False)
note = np.sin(frequency * 2 * np.pi * t)
audio = (note * 32767).astype(np.int16)

with wave.open("test.wav", "w") as f:
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(fs)
    f.writeframes(audio.tobytes())
