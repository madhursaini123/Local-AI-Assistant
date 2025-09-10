import cv2
import pyttsx3 
import time
import numpy as np

modelFile = r"D:\python\Jarvis_AI\models\res10_300x300_ssd_iter_140000.caffemodel"
configFile = r"D:\python\Jarvis_AI\models\deploy.prototxt.txt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

engine = pyttsx3.init()
engine.setProperty('rate', 150)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

last_spoken = time.time()
cooldown = 3
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0,
                                 (300, 300), 
                                 (104.0, 177.0, 123.0))
    
    net.setInput(blob)
    detections = net.forward()
    
    face_count = 0
    
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        if confidence > 0.6:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype('int')
            face_count += 1
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
    label = f"Faces detected: {face_count}"
    cv2.putText(frame, label, (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    if time.time() - last_spoken > cooldown:
        if face_count == 0:
            engine.say("No faces detected")
        elif face_count == 1:
            engine.say("One face detected")
            
        else:
            engine.say(f"{face_count} faces detected")
        engine.runAndWait()
        last_spoken = time.time()
        
    cv2.imshow('Utsi Face Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()