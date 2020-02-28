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

ImageFile.LOAD_TRUNCATED_IMAGES = True

bottleneck_set = 'Xception'
PATH = 'datasets/'
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
train_dir = PATH + 'train'
valid_dir = PATH +'valid'
test_dir = PATH + 'test'
batch_size = 16
epochs = 100

DOG_NAMES = [item[19:-1] for item in sorted(glob(PATH + "train/*/"))]

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

def train_bottleneck(epochs):
	#initial model training without data augmentation
	#saves initial model weights that can be used on data augmeneted training
	bottleneck_features = np.load('bottleneck_features/Dogs' + bottleneck_set + 'Data.npz')
	train_bottle = bottleneck_features['train']
	valid_bottle = bottleneck_features['valid']
	test_bottle = bottleneck_features['test']
	model = Sequential([
		GlobalAveragePooling2D(input_shape = train_bottle.shape[1:]),
		Dense(256, activation = 'relu', kernel_regularizer=regularizers.l2(0.01),),
		Dropout(0.2),
		Dense(133, activation = 'softmax')
	])
	model.summary()
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.bottleneck.hdf5', verbose=1, save_best_only=True)
	history = model.fit(train_bottle, train_targets, 
			validation_data=(valid_bottle, valid_targets),
			epochs=epochs, batch_size=batch_size, callbacks=[checkpointer], verbose=1)
	print_accuracy(model, test_bottle, history)
 
	return model

def train_augmented(epochs):
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
	image_gen_val = ImageDataGenerator(preprocessing_function = xception.preprocess_input, rescale=1./255)
	image_gen_test = ImageDataGenerator(preprocessing_function = xception.preprocess_input, rescale=1./255)

	train_data_gen = image_gen_train.flow_from_directory(batch_size=batch_size,
														directory=train_dir,
														target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
														class_mode='categorical')
	validation_data_gen = image_gen_val.flow_from_directory(batch_size=batch_size,
														directory=valid_dir,
														target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
														class_mode='categorical')

	test_data_gen = image_gen_test.flow_from_directory(batch_size=batch_size,
														directory=test_dir,
														target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
														class_mode='categorical')

	xcep_model = xception.Xception(weights='imagenet', include_top = False, input_shape = (IMAGE_HEIGHT, IMAGE_WIDTH, 3))
	for layer in xcep_model.layers:
		layer.trainable = False

	model = Sequential([
		GlobalAveragePooling2D(input_shape = xcep_model.output_shape[1:]),
		Dense(256, activation = 'relu', kernel_regularizer=regularizers.l2(0.01)),
		Dropout(0.2),
		Dense(133, activation = 'softmax')
	])
	model.summary()

	main_model = Sequential([
		xcep_model,
		model
	])

	main_model.summary()
	main_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.augmented.hdf5', verbose=1, save_best_only=True)
	main_model.fit_generator(train_data_gen,
				validation_data = validation_data_gen,
				epochs=epochs, callbacks=[checkpointer], verbose=1)
	predict = main_model.predict_generator(test_data_gen)
	prediction_labels = np.array([np.argmax(x) for x in predict])
	test_accuracy = np.sum(test_data_gen.labels == prediction_labels)/len(test_data_gen.labels) * 100
	print('Test accuracy: %.4f%%' % test_accuracy)
 
	return main_model

def print_accuracy(model, test_set, history):
	predictions = [np.argmax(model.predict(np.expand_dims(feature, axis=0))) for feature in test_set]
	test_accuracy = 100*np.sum(np.array(predictions)==np.argmax(test_targets, axis=1))/len(predictions)
	print('Test accuracy: %.4f%%' % test_accuracy)
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

def load_targets(path):
	data = load_files(path)
	targets = to_categorical(np.array(data['target']), 133)
	return targets

train_targets = load_targets(train_dir)
valid_targets = load_targets(valid_dir)
test_targets = load_targets(test_dir)

bottleneck_model = train_bottleneck(epochs = epochs)

bottleneck_model.load_weights('saved_models/weights.best.bottleneck.hdf5')

bottleneck_model.save('saved_models/model.h5')

#augmented_training_model = train_augmented(epochs = epochs)



