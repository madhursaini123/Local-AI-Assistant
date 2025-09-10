# --- UTSI VISION MODULE (CLEANED + FIXED WeChat QR + BARCODE RESTORE + SCALING IMPROVEMENT) ---
import cv2
import datetime
import os
import numpy as np
import sounddevice as sd
import time
import webrtcvad
from faster_whisper import WhisperModel
import threading
import asyncio
from edge_tts import Communicate
from fuzzywuzzy import fuzz
from ultralytics import YOLO
import noisereduce as nr
from pyzbar import pyzbar
import webbrowser
from cooldown_manager import CooldownManager

# --- SETUP ---
YOLO_MODEL = YOLO("yolov8n.pt")
SAVE_DIR = "vision_captures"
os.makedirs(SAVE_DIR, exist_ok=True)
os.environ['YOLO_VERBOSE'] = 'False'
cooldown = CooldownManager(cooldown_seconds=5)
SAMPLE_RATE = 16000
CHUNK_MS = 30
CHANNELS = 1
WHISPER_SIZE = "medium.en"
WHISPER_MODEL = WhisperModel(WHISPER_SIZE, device="cpu", compute_type="int8")
vad = webrtcvad.Vad(3)

USE_EDGE_TTS = False
EDGE_VOICE = "en-US-AndrewNeural"

# Load WeChat QRCode Detector (make sure model files exist)
DETECTOR = cv2.wechat_qrcode_WeChatQRCode(
    r"D:\python\Jarvis_AI\models\detect.prototxt",
    r"D:\python\Jarvis_AI\models\detect.caffemodel",
    r"D:\python\Jarvis_AI\models\sr.prototxt",
    r"D:\python\Jarvis_AI\models\sr.caffemodel"
)

# --- TTS ---
async def speak_edge(text):
    try:
        tts = Communicate(text, voice=EDGE_VOICE)
        await tts.save("reply.mp3")
        os.system('start reply.mp3')
    except Exception as e:
        print("Edge-TTS failed:", e)

def speak(text):
    if USE_EDGE_TTS:
        threading.Thread(target=lambda: asyncio.run(speak_edge(text)), daemon=True).start()
    else:
        print("[TTS]:", text)

# --- AUDIO ---
def record_vad(max_sec=2):
    frames = []
    stream = sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype='int16',
        blocksize=int(SAMPLE_RATE * CHUNK_MS / 1000),
    )
    with stream:
        start_time = time.time()
        while time.time() - start_time < max_sec:
            data, _ = stream.read(stream.blocksize)
            if vad.is_speech(data, SAMPLE_RATE):
                frames.append(data)
    if not frames:
        return None
    pcm = b''.join(frames)
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    return nr.reduce_noise(y=audio, sr=SAMPLE_RATE)

def transcribe_audio():
    audio = record_vad()
    if audio is None:
        return ""
    segments, _ = WHISPER_MODEL.transcribe(audio, language="en")
    return "".join(seg.text.strip().lower() for seg in segments if len(seg.text.strip()) > 3)

# --- FUZZY COMMANDS ---
speech_map = {
    "siv": "save", "sape": "save", "snap": "save",
    "captur": "save", "potato": "save",
    "clos": "close", "closs": "close", "shut": "close",
    "shut it": "close", "shut the camera": "close"
}

def smart_command_match(transcribed, command_list, threshold=70):
    best, highest = "", 0
    for cmd in command_list:
        score = fuzz.partial_ratio(transcribed, cmd)
        if score > highest:
            best, highest = cmd, score
    return best if highest > threshold else ""

# --- QR + BARCODE SCANNER ---
def scan_barcodes_and_qrcodes(frame):
    decoded_texts = []

    # Step 1: Resize to simulate higher resolution
    scaled = cv2.resize(frame, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)

    # Step 2: Grayscale
    gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)

    # Step 3: Denoise + sharpen
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(blur, -1, sharpen_kernel)

    # Step 4: Decode with pyzbar
    for img in [gray, sharpened, scaled]:
        barcodes = pyzbar.decode(img)
        for barcode in barcodes:
            x, y, w, h = barcode.rect
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            data = barcode.data.decode("utf-8")
            barcode_type = barcode.type
            cv2.putText(frame, f"{barcode_type}: {data}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            decoded_texts.append(data)

    # Use WeChat QR detection
    try:
        wechat_qrcodes, _ = DETECTOR.detectAndDecode(scaled)
        for qr_text in wechat_qrcodes:
            if qr_text:
                decoded_texts.append(qr_text)
                cv2.putText(frame, f"WeChat QR: {qr_text}", (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    except Exception as e:
        print("⚠️ WeChat QR detection failed:", e)

    return decoded_texts

# --- VISION MAIN ---
def open_webcam():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("❌ Webcam not accessible.")
        return

    print("📷 Say 'save image' or 'close camera'.")
    command = ""
    stop_flag = False
    current_detected = set()
    seen_qr_codes = set()
    object_boxes = {}
    DETECTION_COOLDOWN = 5
    GLOBAL_SPEAK_COOLDOWN = 4
    box_display_timeout = 1.5
    last_detected = {}
    last_spoken = 0
    last_yolo_run = 0
    yolo_interval = 1.0

    def listen():
        nonlocal command, stop_flag
        while not stop_flag:
            cmd = transcribe_audio()
            for key in speech_map:
                if key in cmd:
                    cmd = speech_map[key]
                    break
            if len(cmd.strip()) >= 3:
                print("🗣️ You said:", cmd)
                command = cmd
            time.sleep(1.2)

    threading.Thread(target=listen, daemon=True).start()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()

        # --- QR/BARCODE ---
        qr_results = scan_barcodes_and_qrcodes(frame)
        for qr_text in qr_results:
            if cooldown.is_ready(f"qr:{qr_text}"):
               speak(f"Code says: {qr_text}")
               print("📦 Code:", qr_text)
               if qr_text.startswith("http"):
                  webbrowser.open(qr_text)
                  
        # --- YOLO ---
        if now - last_yolo_run > yolo_interval:
            results = YOLO_MODEL(frame, verbose=False)[0]
            for box in results.boxes:
                if box.conf[0] < 0.5:
                    continue
                cls_id = int(box.cls[0])
                label = YOLO_MODEL.names[cls_id]
                current_detected.add(label)
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                object_boxes[label] = (now, tuple(xyxy))
            last_yolo_run = now

        for label, (ts, (x1, y1, x2, y2)) in list(object_boxes.items()):
            if now - ts > box_display_timeout:
                del object_boxes[label]
                continue
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        for obj in current_detected:
            if cooldown.is_ready(f"object:{obj}"):
                speak(f"Detected {obj}")
                break

        if smart_command_match(command, ["close", "exit", "bye", "stop"]):
            speak("Closing camera.")
            break
        elif smart_command_match(command, ["save", "capture", "snap"]):
            filename = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".png"
            path = os.path.join(SAVE_DIR, filename)
            cv2.imwrite(path, frame)
            print(f"📸 Saved: {path}")
            command = ""
            time.sleep(1)

        cv2.imshow("Utsi Vision", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    stop_flag = True
    cap.release()
    cv2.destroyAllWindows()
