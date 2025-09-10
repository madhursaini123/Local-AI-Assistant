import os
import queue
import sounddevice as sd
import numpy as np
import pyttsx3
from openai import OpenAI
import time
from faster_whisper import WhisperModel
import requests
  
MIC_INDEX = 1 
sd.default.device = (MIC_INDEX, None)      
print(sd.query_devices())
record_second = 20 
sample_rate = 16000
whisper_model = "medium.en"

if not os.getenv("OPENAI_KEY"):
    raise SystemExit("OPENAI_KEY environment variable not set.")

client = OpenAI(api_key=os.getenv("OPENAI_KEY"))

WHISPER_MODEL = "base.en"
print("Loading Whisper model...")
stt_model = WhisperModel(
    WHISPER_MODEL,
    device="cpu",            # ← forces CPU path
    compute_type="int8"      # slightly faster / less RAM
)
print("pyttsx3 engine initializing...")
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 170)

def record_audio(seconds=record_second, fs=sample_rate):
    
    print(f"[Rec] device {MIC_INDEX} → {sd.query_devices(MIC_INDEX)['name']}")
    audio_int16 = sd.rec(int(seconds * fs),
                         samplerate=fs,
                         channels=1,
                         dtype='int16',
                         device=MIC_INDEX)
    sd.wait()
    audio_int16 = audio_int16.flatten()

    max_amp = np.abs(audio_int16).max()
    print("Max amplitude captured:", max_amp)
    if max_amp < 500:                # raise threshold if needed
        print("silence detected, try again.")
        return None

    return audio_int16.astype(np.float32) / 32768.0   # to float32



# ---------- transcribe --------------------------------------------
def transcribe(audio_f32):
    segments, _ = stt_model.transcribe(audio_f32, language="en")
    return "".join([s.text for s in segments]).strip()


def chat(prompt: str) -> str:
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3",
                "prompt": prompt,
                "stream": False
            }
        )
        data = response.json()

        # ✅ Safely get the response field
        if 'response' in data:
            return data['response'].strip()
        else:
            print("⚠ Ollama returned unexpected format:", data)
            return "Sorry, I didn't understand that."

    except Exception as e:
        print("Ollama error:", e)
        return "Sorry, my local brain is not working right now."

def speak(text: str) -> None:
    
    tts_engine.say(text)
    tts_engine.runAndWait()

if __name__ == "__main__":
    print("\n utsi ready. speak after the beep. ctrcl+c to exit.\n")
    try:
        while True:
            audio = record_audio()
            if audio is None:        # silence, restart loop
                continue

            query = transcribe(audio)
            if not query:
             print("(couldn’t recognise speech)")
             continue
            print(f"Query: {query}")
            
            reply = chat(query)
            print(f"utsi: {reply}")
            speak(reply)
            time.sleep(0.3)
            
    except KeyboardInterrupt:
        print("\nExiting...")
            
