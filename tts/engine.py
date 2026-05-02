import edge_tts
import asyncio
import os 
import pygame 
from pyparsing import Optional
import requests 



Voice ="en-US-GuyNeural"

    

async def generate_speech(text, output_file):
    communicate = edge_tts.Communicate(text, Voice)
    await communicate.save(output_file)


def speak(text):
    output_file = "voice/tts/output.mp3"

    asyncio.run(generate_speech(text, output_file))
    pygame.mixer.init()
    pygame.mixer.music.load(output_file)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    pygame.mixer.quit()

