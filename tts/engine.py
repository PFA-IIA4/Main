import edge_tts
import asyncio
import os 
import pygame 

Voice ="en-US-GuyNeural"

async def generate_speech(text, output_file):
    communicate = edge_tts.Communicate(text, Voice)
    await communicate.save(output_file)

def speak(text):
    if not text or not text.strip():
        return

    output_file = os.path.join(os.path.dirname(__file__), "output.mp3")

    asyncio.run(generate_speech(text, output_file))
    # Pre-init the mixer to use a larger buffer to prevent ALSA underruns
    pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=4096)
    pygame.mixer.init()
    pygame.mixer.music.load(output_file)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    pygame.mixer.quit()

