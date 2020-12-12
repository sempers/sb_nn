import cv2
import sys
import tensorflow as tf
from collections import deque
import threading
import numpy as np
import time
import datetime
from EmotionClassifier import EmotionClassifier

faces_queue = deque(maxlen=1)               #очередь лиц
classes_queue = deque(maxlen=1) 			#готовые классы

cascPath = "./haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
video_capture = cv2.VideoCapture(0)

STOP_THREAD = False

classifier = EmotionClassifier()

#цикл распознавания эмоции
def recognize_emotion_cycle():
	while not STOP_THREAD:
		if len(faces_queue) > 0:
			face = faces_queue.pop()
			class_name, delta  = classifier.predict(face)
			print(f"INFERENCE MADE in {delta} seconds. PREDICTED CLASS: {class_name}")
			classes_queue.append(class_name)
		time.sleep(0.1)

threading.Thread(target=recognize_emotion_cycle).start()

#основной цикл
while True:
	ret, frame = video_capture.read()
	
	if frame is None:
		print("No web camera found! Application terminates")
		break
		
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = faceCascade.detectMultiScale(
		gray,
		scaleFactor=1.1,
		minNeighbors=5,
		minSize=(30, 30),
		flags=cv2.CASCADE_SCALE_IMAGE
	)

	# Рисуем рамку с эмоцией
	o = 25 # Статический оффсет, т.к. opencv делает слишком сильный кроп, не соотв. датасету 

	for (x, y, w, h) in faces:
		cv2.rectangle(frame, (x-o, y-o), (x+w+o, y+h+o), (0, 255, 0), 2)
		if len(classes_queue) > 0:
			class_name = classes_queue[0]
		else:
			class_name='Model offline'
		cv2.putText(frame, class_name, (x-o, y-o-10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (36,255,12))
		try:
			face_img = cv2.resize(frame[x-o:x+w+o, y-o:y+h+o], (224, 224), interpolation=cv2.INTER_AREA)
			faces_queue.append(face_img)
		except: pass
	
	#Рисуем кадр
	cv2.imshow('Video', frame)

	key = cv2.waitKey(1) & 0xFF
	if key == ord('q'):
		break
	elif key == ord('p'):
		cv2.imwrite(f"scr-{datetime.datetime.now().strftime('%Y-%M-%d-%H-%m-%s')}.jpg", frame)
		print("screenshot saved")

# Освобождаем ресурсы
STOP_THREAD = True
video_capture.release()
cv2.destroyAllWindows()