import asyncio
import os
import pyttsx3

USE_EDGE_TTS = True
EDGE_VOICE = "en-US-AndrewNeural"

engine = pyttsx3.init()
engine.setProperty("rate", 170)

async def speak_edge(text):
    try:
        from edge_tts import Communicate
        out_path = "reply.mp3"
        tts = Communicate(text, voice=EDGE_VOICE)
        await tts.save(out_path)
        os.system(f'start "" "{out_path}"')  # Windows quick play
    except Exception as e:
        print("Edge‑TTS failed:", e)
        speak_local(text)

def speak_local(text):
    engine.say(text)
    engine.runAndWait()

def speak(text):
    if USE_EDGE_TTS:
        try:
            asyncio.run(speak_edge(text))
        except Exception as e:
            print("Edge‑TTS run failed:", e)
            speak_local(text)
    else:
        speak_local(text)
