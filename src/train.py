import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
from PIL import ImageFile 
from sklearn.datasets import load_files 
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint  
from tensorflow.keras import regularizers
from tensorflow.keras.applications import xception
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import History

ImageFile.LOAD_TRUNCATED_IMAGES = True

bottleneck_set = 'Xception'
PATH = 'datasets/'
IMAGE_HEIGHT = 300
IMAGE_WIDTH = 300
train_dir = PATH + 'train'
valid_dir = PATH +'valid'
test_dir = PATH + 'test'
batch_size = 16
epochs = 10

DOG_NAMES = [item[19:-1] for item in sorted(glob(PATH + "train/*/"))]
NUM_CLASSES = len(DOG_NAMES)

def load_tensors(path):
	data = load_files(path)
	dog_files = np.array(data['filenames'])
	dog_tensors = [path_to_tensor(img_path) for img_path in tqdm(dog_files)]
	dog_tensors = np.vstack(dog_tensors)
	dog_tensors = dog_tensors/255.0
	return dog_tensors

def path_to_tensor(img_path):
	img = image.load_img(img_path, target_size=(IMAGE_HEIGHT, IMAGE_WIDTH))
	x = image.img_to_array(img)
	return np.expand_dims(x, axis=0)

def load_targets(path):
	data = load_files(path)
	targets = to_categorical(np.array(data['target']), NUM_CLASSES)
	return targets

def get_bottleneck_model():
	model = Sequential([
		GlobalAveragePooling2D(input_shape = xception.Xception(weights='imagenet', include_top=False).output_shape[1:]),
		Dense(256, activation = 'relu', kernel_regularizer=regularizers.l2(0.01),),
		Dropout(0.2),
		Dense(133, activation = 'softmax')
	])
	model.summary()
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

def train_bottleneck(model, epochs):
	#initial model training without data augmentation
	#saves initial model weights that can be used on data augmeneted training
	bottleneck_features = np.load('bottleneck_features/Dogs' + bottleneck_set + 'Data.npz')
	train_bottle = bottleneck_features['train']
	valid_bottle = bottleneck_features['valid']
	test_bottle = bottleneck_features['test']
	checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.bottleneck.hdf5', verbose=1, save_best_only=True)
	history = model.fit(train_bottle, train_targets, 
			validation_data=(valid_bottle, valid_targets),
			epochs=epochs, batch_size=batch_size, callbacks=[checkpointer], verbose=1)
	print_test_accuracy(model, test_tensors)
	accuracy_history(model, history)
	return model

def get_augmented_model():
	xcep_model = xception.Xception(weights='imagenet', include_top = False, input_shape = (IMAGE_HEIGHT, IMAGE_WIDTH, 3))
	for layer in xcep_model.layers:
		layer.trainable = False
	model = Sequential([
		GlobalAveragePooling2D(input_shape = xcep_model.output_shape[1:]),
		Dense(512, activation = 'relu', kernel_regularizer=regularizers.l2(0.001)),
		Dropout(0.1),	
		Dense(256, activation = 'relu', kernel_regularizer=regularizers.l2(0.001)),
		Dropout(0.1),
		Dense(NUM_CLASSES, activation = 'softmax')
	])
	model.summary()

	main_model = Sequential([
		xcep_model,
		model
	])

	main_model.summary()
	main_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
 
	return main_model

def train_augmented(model, epochs):
	#data preprocess and train data augmentation.
	image_gen_train = ImageDataGenerator(
					preprocessing_function = xception.preprocess_input,
					rescale=1./255,
					rotation_range=20,
					width_shift_range=.15,
					height_shift_range=.15,
					horizontal_flip=True,
					zoom_range=0.3,
					fill_mode = 'nearest'
	)

	train_data_gen = image_gen_train.flow_from_directory(
					batch_size=batch_size,
					directory=train_dir,
					target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
					class_mode='categorical')

	#load valid and test data without augmentation and without any generators.
	valid_tensors = load_tensors(valid_dir)
	test_tensors = load_tensors(test_dir)

	#train model
	checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.augmented.hdf5', verbose=1, save_best_only=True)
	history = model.fit(train_data_gen,
			validation_data = (valid_tensors, valid_targets),
			epochs=epochs, callbacks=[checkpointer], verbose=1)
	print_test_accuracy(model, test_tensors)
	accuracy_history(model, history)

def accuracy_history(model, history):
	acc = history.history['accuracy']
	val_acc = history.history['val_accuracy']
	loss = history.history['loss']
	val_loss = history.history['val_loss']
	epochs_range = range(epochs)
	plt.figure(figsize=(8, 8))
	plt.subplot(1, 2, 1)
	plt.plot(epochs_range, acc, label='Training Accuracy')
	plt.plot(epochs_range, val_acc, label='Validation Accuracy')
	plt.legend(loc='lower right')
	plt.title('Training and Validation Accuracy')
	plt.subplot(1, 2, 2)
	plt.plot(epochs_range, loss, label='Training Loss')
	plt.plot(epochs_range, val_loss, label='Validation Loss')
	plt.legend(loc='upper right')
	plt.title('Training and Validation Loss')
	plt.show()

def print_test_accuracy(model, test_tensors):
	predictions = [np.argmax(model.predict(np.expand_dims(feature, axis=0))) for feature in test_tensors]
	test_accuracy = 100*np.sum(np.array(predictions)==np.argmax(test_targets, axis=1))/len(predictions)
	print('Test accuracy: %.4f%%' % test_accuracy)

train_targets = load_targets(train_dir)
valid_targets = load_targets(valid_dir)
test_targets = load_targets(test_dir)
test_tensors = load_tensors(test_dir)
if __name__ == "__main__":

	# bottleneck_model = get_bottleneck_model()
	# train_bottleneck(bottleneck_model, epochs)
	# bottleneck_model.save('saved_models/model_bottleneck_features.h5')

	# augmented_model = get_augmented_model() 
	# train_augmented(augmented_model, epochs)
	# augmented_model.save('saved_models/model_data_augmentation.h5')

	model = tf.keras.models.load_model('saved_models/model_data_augmentation.h5')
	print_test_accuracy(model, test_tensors)

