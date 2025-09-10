import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array

class EmotionDetector:
    def __init__(self, model_path=r'D:\python\Jarvis_AI\models\fer2013_mini_XCEPTION.102-0.66.hdf5'):
        self.model = load_model(model_path, compile = False)
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
    def detect_emotion(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        emotions= []
        
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