import numpy as np
import pyaudio
import time

# Parameters
volume = 0.5      # range [0.0, 1.0]
fs = 44100        # sampling rate, Hz
duration = 1.0    # seconds
f = 440.0         # sine frequency, Hz (A4 note)

# Generate samples
samples = (np.sin(2 * np.pi * np.arange(fs * duration) * f / fs)).astype(np.float32)

# Initialize PyAudio
p = pyaudio.PyAudio()

# Open stream (use default output device, which should be I2S if configured)
stream = p.open(format=pyaudio.paFloat32,
                channels=1,
                rate=fs,
                output=True)

# Play sound
stream.write(volume * samples)

# Cleanup
stream.stop_stream()
stream.close()
p.terminate()
