import cv2
import sys
import tensorflow as tf
import keras
from collections import deque
import threading
import numpy as np
import time
import datetime

model = keras.models.load_model("model.h5")
print("Model loaded")

CLASSES = ["anger", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise", "uncertain"]
CLASSES_MAPPING = {i: s for i, s in enumerate(CLASSES)}

faces_queue = deque(maxlen=1)               #очередь лиц
classes_queue = deque(maxlen=1) 			#готовые классы

cascPath = "./haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
video_capture = cv2.VideoCapture(0)

#препроцессинг
def preprocess_image(x):
    x_temp = x.astype(np.float16)
    #x_temp = x_temp[..., ::-1] #BGR уже приходит из OpenCV
    x_temp[..., 0] -= 91.4953
    x_temp[..., 1] -= 103.8827
    x_temp[..., 2] -= 131.0912
    return np.expand_dims(x_temp, 0)

STOP_THREAD = False

#цикл распознавания эмоции
def recognize_emotion_cycle():
    while not STOP_THREAD:
        if len(faces_queue) > 0:
            face = faces_queue.pop()
            now = datetime.datetime.now()
            y = model.predict(preprocess_image(face))[0]
            delta = (datetime.datetime.now() - now).total_seconds()
            class_name = CLASSES_MAPPING[np.argmax(y)]
            print(f"INFERENCE MADE in {delta} seconds. PREDICTED CLASS: {class_name}")
            classes_queue.append(class_name)
        time.sleep(0.1)

threading.Thread(target=recognize_emotion_cycle).start()

#основной цикл
while True:
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Рисуем рамку с эмоцией
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        if len(classes_queue) > 0:
            class_name = classes_queue[0]
        else:
            class_name='Not recognized'
        cv2.putText(frame, class_name, (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (36,255,12))
        try:
            face_img = cv2.resize(frame[x:x+w, y:y+h], (224, 224), interpolation=cv2.INTER_AREA)
            faces_queue.append(face_img)
        except: pass
	
	#Рисуем кадр
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождаем ресурсы
STOP_THREAD = True
video_capture.release()
cv2.destroyAllWindows()