import keras
import datetime
import numpy as np

class EmotionClassifier:
	def __init__(self):
		self.model = keras.models.load_model("model.h5")
		self.CLASSES = ["anger", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise", "uncertain"]
		self.CLASSES_MAPPING = {i: s for i, s in enumerate(self.CLASSES)}
		print("NN model loaded")    

	#препроцессинг
	def preprocess_image(self, x):
		x_temp = x.astype(np.float16)
		#x_temp = x_temp[..., ::-1] #BGR уже приходит из OpenCV
		x_temp[..., 0] -= 91.4953
		x_temp[..., 1] -= 103.8827
		x_temp[..., 2] -= 131.0912
		return np.expand_dims(x_temp, 0)

	def predict(self, img):
		now = datetime.datetime.now()
		img = self.preprocess_image(img)
		y = self.model.predict(img)[0]
		delta = (datetime.datetime.now() - now).total_seconds()
		return self.CLASSES_MAPPING[np.argmax(y)], delta
