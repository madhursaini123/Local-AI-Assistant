import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import pyttsx3
import time

# Optional: simple cooldown manager
class CooldownManager:
    def __init__(self):
        self.cooldowns = {}

    def is_ready(self, key, cooldown_seconds):
        current_time = time.time()
        if key not in self.cooldowns or current_time - self.cooldowns[key] >= cooldown_seconds:
            self.cooldowns[key] = current_time
            return True
        return False

# Text-to-speech
def speak(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 160)
    engine.say(text)
    engine.runAndWait()

class EmotionDetector:
    def __init__(self, model_path=r'D:\python\Jarvis_AI\models\fer2013_mini_XCEPTION.102-0.66.hdf5'):
        self.model = load_model(model_path, compile=False)
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
    def detect_emotion(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        emotions = []
        
        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (64, 64))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            
            preds = self.model.predict(roi, verbose=0)[0]
            emotion = self.emotion_labels[np.argmax(preds)]
            emotions.append((emotion, (x, y, w, h)))
            
        return emotions

if __name__ == "__main__":
    emotion_detector = EmotionDetector()
    cooldowns = CooldownManager()

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        emotions = emotion_detector.detect_emotion(frame)

        for emotion, (x, y, w, h) in emotions:
            label = f"Emotion: {emotion}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            if cooldowns.is_ready(f"emotion_{emotion}", 10):
                speak(f"Detected emotion: {emotion}")

        cv2.imshow("Emotion Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
