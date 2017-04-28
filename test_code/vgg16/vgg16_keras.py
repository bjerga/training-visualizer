import pickle
from time import time
from os import listdir
from os.path import join, dirname

import numpy as np

import keras.backend as K
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image

# import callbacks for visualizing
from custom_keras.callbacks import NetworkSaver, TrainingProgress, LayerActivations, SaliencyMaps, DeepVisualization, Deconvolution

# find path to save networks and results
save_path = dirname(__file__)

# find path to imagenet URLs
imagenet_path = save_path.replace('vgg16', 'imagenet')
train_data_path = join(imagenet_path, 'train_data')
test_data_path = join(imagenet_path, 'test_data')
val_data_path = join(imagenet_path, 'val_data')

train_data_names = listdir(train_data_path)
train_data_amount = len(train_data_names)

MEAN_VALUES = np.array([103.939, 116.779, 123.68])

with open(join(imagenet_path, 'wnid_index_map.pickle'), 'rb') as f:
	wnid_index_map = pickle.load(f)


def create_model(input_shape):
	# define model
	# model = VGG16(include_top=True, input_shape=input_shape)
	model = VGG16()
	model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
	
	print('\nModel successfully created')
	
	return model


def train(model, no_of_epochs=10):
	# train top layers of model (self-defined layers)
	print('\n\nCommence VGG16 model training\n')
	
	# initialize custom callbacks
	# neurons_to_visualize = [(-1, 0), (-1, 1), (-1, 2)]
	# custom_callbacks = [NetworkSaver(save_path), TrainingProgress(save_path), LayerActivations(save_path), SaliencyMaps(save_path),
	# 					DeepVisualization(save_path, neurons_to_visualize,
	# 									  2500.0, 500, l2_decay=0.0001, blur_interval=4, blur_std=1.0),
	# 					Deconvolution(save_path, feat_map_layer_no=3, feat_map_amount=3)]
	
	# batch_size = 256
	batch_size = 2
	steps_per_epoch = train_data_amount / batch_size
	
	print('Steps per epoch:', steps_per_epoch)
	
	model.fit_generator(generator=data_generator(batch_size), steps_per_epoch=steps_per_epoch, epochs=30, verbose=1)
	
	print('\nCompleted VGG16 model training\n')
	
	return model

def data_generator(batch_size):
	
	zeroes = np.zeros((1, 1000))
	while True:
		random_img_names = np.random.choice(train_data_names, batch_size, replace=False)
		for img_name in random_img_names:
		
			# x_data = np.array([preprocess_data(load_image(train_data_path, img_name), (224, 224)) for img_name in random_img_names])
			x_data = preprocess_data(load_image(train_data_path, img_name), (224, 224))
			x_data = np.expand_dims(x_data, 0)
			# y_data = [wnid_index_map[img_name] for img_name in random_img_names]
			y_data = zeroes.copy()
			y_data[0][42] = 1.0

			yield x_data, y_data

def load_image(img_path, img_name):
	img = image.load_img(join(img_path, img_name))
	return image.img_to_array(img)

def preprocess_data(img_array, target_size):
	
	# change size of image
	img = image.array_to_img(img_array)
	height_weight_tuple = (target_size[1], target_size[0])
	if img.size != height_weight_tuple:
		img = img.resize(height_weight_tuple)
		img_array = image.img_to_array(img)
	
	# subtract mean values
	if K.image_data_format() == 'channels_last':
		img_array -= MEAN_VALUES.reshape((1, 1, 3))
	else:
		img_array -= MEAN_VALUES.reshape((3, 1, 1))
	
	return img_array


def postprocess_data(img_array):
	img_array += MEAN_VALUES.reshape((1, 1, 3))
	return img_array


def main():
	# create a model, then train and test it.
	
	start_time = time()
	
	if K.image_data_format() == 'channels_last':
		input_shape = (224, 224, 3)
	else:
		input_shape = (3, 224, 224)

	model = create_model(input_shape)
	
	model = train(model)
	
	print('This took %.2f seconds' % (time() - start_time))


main()
