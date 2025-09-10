import cv2
import numpy as np
import webbrowser
from pyzbar.pyzbar import decode
from utsiskills.utils import speak

# ✅ Load WeChat QR Detector
DETECTOR = cv2.wechat_qrcode_WeChatQRCode(
    r"D:\python\Jarvis_AI\models\detect.prototxt",
    r"D:\python\Jarvis_AI\models\detect.caffemodel",
    r"D:\python\Jarvis_AI\models\sr.prototxt",
    r"D:\python\Jarvis_AI\models\sr.caffemodel"
)

seen_codes = set()

def scan_barcodes_and_qrcodes(frame):
    decoded_texts = []

    # ⏫ Upscale for better detection
    scaled = cv2.resize(frame, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)

    # 🎯 Barcode & QR with pyzbar (best for barcodes)
    barcodes = decode(scaled)
    for barcode in barcodes:
        x, y, w, h = barcode.rect
        data = barcode.data.decode("utf-8")
        btype = barcode.type
        if data not in decoded_texts:
            decoded_texts.append(data)
            cv2.rectangle(frame, (x//2, y//2), ((x + w)//2, (y + h)//2), (0, 255, 0), 2)
            cv2.putText(frame, f"{btype}: {data}", (x//2, y//2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 🎯 QR with WeChat (optional, more robust for distorted/far QR)
    try:
        wechat_qrcodes, _ = DETECTOR.detectAndDecode(scaled)
        for qr_text in wechat_qrcodes:
            if qr_text and qr_text not in decoded_texts:
                decoded_texts.append(qr_text)
                cv2.putText(frame, f"WeChat QR: {qr_text}", (30, 30 + 25 * len(decoded_texts)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    except Exception as e:
        print("⚠️ WeChat QR detection failed:", e)

    return decoded_texts

def open_webcam():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("❌ Webcam not accessible.")
        return

    print("🔍 QR/Barcode Scanner Active — Press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = scan_barcodes_and_qrcodes(frame)
        for text in results:
            if text not in seen_codes:
                seen_codes.add(text)
                speak(f"Code says: {text}")
                print("📦 Code:", text)
                if text.startswith("http"):
                    webbrowser.open(text)

        cv2.imshow("Utsi QR/Barcode Scanner", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    open_webcam()
