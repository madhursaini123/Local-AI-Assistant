"""
Utsi ‑ Local Voice Assistant
----------------------------
Mic  ➜  WebRTC‑VAD  ➜ Whisper STT ➜ LLaMA‑3 via Ollama ➜ Edge TTS (or pyttsx3)
"""
import os, asyncio, time, json, requests
import numpy as np
import sounddevice as sd
import webrtcvad
import pyttsx3
from faster_whisper import WhisperModel
from dotenv import load_dotenv
import json
import sys
import threading
import subprocess, webbrowser, pathlib, re
from pathlib import Path
from utsiskills.commands import maybe_run_command
from skills_loader import load_skills, run_skill_if_match
from deep_translator import GoogleTranslator
from utsiskills.utils import speak  # use shared TTS
from brain.intelligence import recall_memory
from brain.last_topic import save_last_topic, get_last_topic, TOPIC_PATH, get_best_context_from_session
from utsiskills import reminder
from utsiskills.rag_engine2 import add_memory, search_memory
from utsiskills.reminder import check_upcoming_reminders
from sentence_transformers import SentenceTransformer, util
import warnings
import torch
from dotenv import load_dotenv
import os

warnings.filterwarnings("ignore", category=FutureWarning)
model = SentenceTransformer("all-MiniLM-L6-v2")
sys.path.append(str(Path(__file__).parent))
load_skills()
MEMORY_PATH = Path(r"D:\python\memory\session.json")

def think(user_input, reply):
    """Store conversation turn into session memory (last 20 entries)."""
    memory = []
    if MEMORY_PATH.exists():
        try:
            memory = json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    memory.append({"user": user_input, "utsi": reply})
    memory = memory[-20:]  # keep last 20 turns
    MEMORY_PATH.write_text(json.dumps(memory, indent=2), encoding="utf-8")

# --------------------------------------------------
# 1.  CONFIGURATION
# --------------------------------------------------
MIC_INDEX      = 1
CHANNELS = 1  # your headset mic device index
SAMPLE_RATE    = 16_000
CHUNK_MS       = 30         # VAD frame size (10/20/30 ms)
RECORD_WINDOW  = 10         # seconds window to listen each turn
WHISPER_SIZE   = "large-v3"  # tiny.en, base.en, medium.en ...
OLLAMA_MODEL   = "deepseek-r1"   # must be pulled with ollama first
USE_EDGE_TTS   = True       # False → fallback to pyttsx3 only
EDGE_VOICE     = "en-US-AndrewNeural"

sd.default.device = (None, None)

# --------------------------------------------------
# 2.  INITIALISE MODELS
# --------------------------------------------------
print("Loading Whisper model …")
stt = WhisperModel(
    WHISPER_SIZE,
    device="cuda",
    compute_type="float16"
)

print(torch.cuda.is_available())  # Should be True
print(torch.cuda.get_device_name(0))
print("Initialising TTS …")
tts_local = pyttsx3.init()
tts_local.setProperty("rate", 170)

vad = webrtcvad.Vad(3)      # aggressiveness 0‑3; 2 = good balance

# --------------------------------------------------
# 3.  AUDIO RECORDING WITH VAD
# --------------------------------------------------
def record_vad(max_sec=RECORD_WINDOW, fs=SAMPLE_RATE):
    """Record up to max_sec seconds, keep only voiced chunks."""
    frames = []
    num_bytes = int(fs * CHUNK_MS / 1000) * 2  # 16‑bit mono
    stream = sd.RawInputStream(
        samplerate=fs,
        channels=CHANNELS ,
        dtype="int16",
        blocksize=int(fs * CHUNK_MS / 1000),
        device=MIC_INDEX
    )
    with stream:
        start_t = time.time()
        while True:
            data, _ = stream.read(stream.blocksize)
            if vad.is_speech(data, fs):
                frames.append(data)
            if time.time() - start_t > max_sec:
                break
    if not frames:
        return None
    # concatenate & convert to float32
    voiced_pcm = b"".join(frames)
    audio = np.frombuffer(voiced_pcm, dtype=np.int16).astype(np.float32) / 32768.0
    return audio

# --------------------------------------------------
# 4.  ASYNC EDGE TTS (fallback to pyttsx3)
# --------------------------------------------------
async def speak_edge(text):
    try:
        from edge_tts import Communicate
        out_path = "reply.mp3"
        tts = Communicate(text, voice=EDGE_VOICE)
        await tts.save(out_path)
        os.system(f'start "" "{out_path}"')     # Windows quick play
    except Exception as e:
        print("Edge‑TTS failed:", e)
        speak_local(text)

def speak_local(text):
    tts_local.say(text)
    tts_local.runAndWait()

def speak(text):
    if USE_EDGE_TTS:
        asyncio.run(speak_edge(text))
    else:
        speak_local(text)

def open_browser(url="https://www.google.com/"):
    """Open a URL in the default web browser."""
    webbrowser.open(url)
    
def open_youtube(query=None):
    """Open a YouTube search for the given query."""
    if query:
        webbrowser.open(f"https://www.youtube.com/results?search_query={query}")
        
    else:
        webbrowser.open("https://www.youtube.com/")

def open_chrome():
    chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    subprocess.Popen([chrome_path])

def open_file(path):
    p = pathlib.Path(path)
    if p.exists():
        os.startfile(str(p))
        
    else:
        speak(f"File not found: {p}")
        
    # Detect language using GoogleTranslator or set a default
    try:
        language = GoogleTranslator(source='auto', target='en').detect(text)
    except Exception:
        language = "en"
    print(f"detected language: {language}")
    
    if language == "hi":
        try:
            original_text = text
            text = GoogleTranslator(source='auto', target='en').translate(text)
            print(f"translated: {original_text} to {text}")
        except Exception as e:
            print("Translation error:", e)
'''
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

def get_last_topic(vague_input=None):
    if not vague_input or len(vague_input.split()) < 2:
        return None

    if not MEMORY_PATH.exists():
        return None

    try:
        memory = json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
        last_user_inputs = [turn["user"] for turn in memory if len(turn["user"].split()) > 3]
        if not last_user_inputs:
            return None

        # Encode all past questions + current vague input
        past_embeddings = model.encode(last_user_inputs, convert_to_tensor=True)
        input_embedding = model.encode(vague_input, convert_to_tensor=True)

        # Compute similarities
        scores = util.pytorch_cos_sim(input_embedding, past_embeddings)[0]
        best_idx = scores.argmax().item()

        return last_user_inputs[best_idx]

    except Exception as e:
        print("🛑 Context error:", e)
        return None
'''
# --------------------------------------------------
# 5.  LLaMA 3 chat via Ollama
# --------------------------------------------------

chat_history = []

def llama_chat_with_history(user_input):
    global chat_history
    try:
        chat_history.append({"role": "user", "content": user_input})

        rasp = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": chat_history,
                "options": {"temperature": 0.7}
            },
            timeout=60
        )

        data = rasp.json()
        reply = ""

        if "message" in data and "content" in data["message"]:
            reply = data["message"]["content"].strip()
        elif "response" in data:
            reply = data["response"].strip()
        else:
            reply = "Sorry, no valid response from Ollama."

        chat_history.append({"role": "assistant", "content": reply})
        return reply

    except Exception as e:
        print("🛑 DeepSeek chat error:", e)
        return "Sorry, I couldn't reach DeepSeek right now."

# --------------------------------------------------
# 6.  MAIN LOOP
# --------------------------------------------------
print("\n🟢 Utsi is ready — speak after the beep (Ctrl+C to quit).\n")
try:
    while True:                                      # <-- continuous loop
        # --- Ask for input mode ---
        mode = input("🔘 Type 's' to speak or 't' to type: ").strip().lower()

        if mode == "s":
            audio = record_vad()
            if audio is None:
                continue
            segments, _ = stt.transcribe(audio, language="en")
            text = " ".join(seg.text for seg in segments).strip().lower()

        elif mode == "t":
            text = input("🧑 You: ").strip().lower()

        else:
            print("Invalid input. Please type 's' or 't'.")
            continue
        
        if not text or len(text) < 3:
           print("🎤 Try again or type your question:")
           text = input("🧑 You: ")


        print("👤 You:", text)

# ------ check commands BEFORE LLM ------
        if maybe_run_command(text):   
         continue

        if run_skill_if_match(text):  
         continue

        if reminder.can_handle(text):
         reminder.handle(text)
         continue

        if "recall" in text or "remember" in text:
          keyword = text.split("recall")[-1].strip() or text.split("remember")[-1].strip()
          reply = recall_memory(keyword)
          print("🧠 Recall:", reply)
          speak(reply)
          continue
       
        # ------ LLM reply ------
        reply = llama_chat_with_history(text)
        print("🤖 Utsi:", reply)
        think(text, reply)  # store conversation turn
        add_memory(text, reply)
        speak(reply)
        
except KeyboardInterrupt:
    print("\nExiting…")
except Exception as e:
    print("An error occurred:", e)
