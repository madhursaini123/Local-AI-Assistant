'''
import cv2
import numpy as np
import logging
from textblob import TextBlob
from difflib import SequenceMatcher
from paddleocr import PaddleOCR

# -----------------------------------------------------------------
# Check for CUDA
gpu_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
print("GPU available:", gpu_available)

# -----------------------------------------------------------------
# Silence paddleocr's own logger
logging.getLogger('paddleocr').setLevel(logging.ERROR)

# -----------------------------------------------------------------
# Initialise ONE OCR model (only once!)
ocr = PaddleOCR(
    lang='en',
    use_angle_cls=True,          # do angle classification
    use_gpu=gpu_available,       # will be ignored if not compiled with CUDA
)
# ------------------- Text Processing Utilities ------------------- #
def correct_text(texts):
    """Spell‑correct each detected string."""
    return [str(TextBlob(t).correct()) for t in texts]

def is_similar(a, b, threshold=0.85):
    """Return True if strings a and b have similarity > threshold."""
    return SequenceMatcher(None, a, b).ratio() > threshold

def clean_detected_text(text_list):
    """Remove duplicates, short strings and very‑similar texts."""
    filtered = []
    for text in text_list:
        text = text.strip()
        if len(text) < 3:               # skip tiny garbage
            continue
        if any(is_similar(text, f) for f in filtered):
            continue
        filtered.append(text)
    return filtered

# ------------------- Image Preprocessing ------------------- #
def preprocess(frame):
    """Sharpen + Otsu‑threshold the frame for better OCR."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Simple sharpening kernel
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]], dtype=np.float32)
    sharp = cv2.filter2D(gray, -1, kernel)

    # Otsu binary threshold
    _, thresh = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

# ------------------- OCR Detection ------------------- #
def detect_text_with_paddleocr(frame):
    """
    Use the global `ocr` instance to detect text.
    Returns:
        output_frame : frame with bounding boxes drawn
        text_data    : list of (text, confidence) tuples
    """
    # ---------------------------------------------------------
    # 1. Pre‑process (optional – you can skip if you prefer raw)
    processed = preprocess(frame)

    # 2. Run OCR (no `cls=` argument!)
    results = ocr.predict(processed)          # <— this line fixed the crash

    # ---------------------------------------------------------
    # 3. Draw results
    output_frame = frame.copy()
    text_data = []

    for line in results:
        # Each `line` is a list: [[box], [text, conf]]
        if len(line) != 2:
            continue
        box, (text, confidence) = line
        box = np.array(box, dtype=np.int32)

        # Draw bounding box & label
        cv2.polylines(output_frame, [box], True, (0, 255, 0), 2)
        cv2.putText(output_frame,
                    f'{text} ({confidence:.2f})',
                    (box[0][0], box[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1)

        text_data.append((text, confidence))

    return output_frame, text_data


# ------------------- Main OCR Camera Loop ------------------- #
def run_paddleocr_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Webcam not accessible.")
        return

    print("🔍 OCR Scanner Active — Press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection
        output_frame, text_data = detect_text_with_paddleocr(frame)

        # Post‑process the detected texts
        only_texts = [t for t, _ in text_data]
        only_texts = correct_text(only_texts)
        only_texts = clean_detected_text(only_texts)

        if only_texts:
            print("Detected:", only_texts)

        # Show live preview
        cv2.imshow("Utsi OCR Scanner", output_frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ------------------- Run Script ------------------- #
if __name__ == "__main__":
    run_paddleocr_camera()
'''