from tensorflow import keras
import numpy as np
from glob import glob
from tensorflow.keras.applications import resnet50
from tensorflow.keras.applications import xception
from PIL import ImageFile
from tensorflow.keras.preprocessing import image

ImageFile.LOAD_TRUNCATED_IMAGES = True

DOG_NAMES = [item[19:-1] for item in sorted(glob("datasets/train/*/"))]
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

def get_model():
	return keras.models.load_model('saved_models/model.h5')

def preprocess_image(img_path):
	img = image.load_img(img_path, target_size = (IMAGE_HEIGHT, IMAGE_WIDTH))
	img = image.img_to_array(img)
	img = np.expand_dims(img , axis = 0)
	return img

def bottleneck_predict_image(model, img_path):
	if dog_classifier(img_path):
		img = preprocess_image(img_path)
		img = xception.preprocess_input(img)
		prediction = xception.Xception(weights='imagenet', include_top=False).predict(img)
		predicted_vector = model.predict(prediction)
		return DOG_NAMES[np.argmax(predicted_vector)]
	else:
		return "not a doggie"

def dog_classifier(img_path):
	img = preprocess_image(img_path)
	img = resnet50.preprocess_input(img)
	prediction = np.argmax(resnet50.ResNet50(weights='imagenet').predict(img))
	#indicies of dogs in pre-trained model
	return ((prediction <= 268) & (prediction >= 151))