import pickle
from math import ceil
from time import time
from os import listdir
from os.path import join, dirname

import numpy as np

import keras.backend as K
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.utils.np_utils import to_categorical

# import callbacks for visualizing
from custom_keras.callbacks import CustomCallbacks

# find path to imagenet URLs
imagenet_path = '/media/anniea/Mikal My Book/ImageNet'
# imagenet_path = 'E:/ImageNet'
train_data_path = join(imagenet_path, 'ILSVRC2012_img_train')
test_data_path = join(imagenet_path, 'ILSVRC2012_img_test')
val_data_path = join(imagenet_path, 'ILSVRC2012_img_val')

train_img_directories = listdir(train_data_path)
train_data_amount = 1281167

batch_size = 64
steps_per_epoch = ceil(train_data_amount / batch_size)

MEAN_VALUES = np.array([103.939, 116.779, 123.68])

if K.image_data_format() == 'channels_last':
	input_shape = (224, 224, 3)
else:
	input_shape = (3, 224, 224)

with open(join(imagenet_path, 'wnid_index_map.pickle'), 'rb') as f:
	wnid_index_map = pickle.load(f)


def create_model(weights=None, untrainable=False):
	# define model
	model = VGG16(include_top=True, weights=weights, input_shape=input_shape)
	
	if untrainable:
		for i in range(len(model.layers)):
			model.layers[i].trainable = False
	
	model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

	print('\nModel successfully created')
	
	return model


def train(model, no_of_epochs=1):
	# train top layers of model (self-defined layers)
	print('\n\nCommence VGG16 model training\n')

	# initialize custom callbacks, use dirname to find path to save networks and results
	callbacks = CustomCallbacks(dirname(__file__), preprocess_data, postprocess_data, base_interval=1)
	callbacks.register_network_saver()
	callbacks.register_training_progress()
	callbacks.register_layer_activations()
	callbacks.register_saliency_maps()
	callbacks.register_deconvolution_network(3, feat_map_amount=20)
	callbacks.register_deconvolution_network(10, feat_map_nos=[2, 25, 26, 37, 69, 84, 92, 152, 171, 176, 187, 196, 239, 245])
	liste = [(21, i) for i in range(0, 4096, 512)]
	callbacks.register_deep_visualization([(22, 76)], 2500.0, 500, l2_decay=0,
										  blur_interval=4, blur_std=0.5, abs_contribution_percentile=90)
	callbacks.register_visualization_snapshot('/home/anniea/Code/results/vgg_results')

	print('Steps per epoch:', steps_per_epoch)

	model.fit_generator(generator=data_generator(batch_size), steps_per_epoch=steps_per_epoch, epochs=no_of_epochs,
						verbose=1, max_q_size=5, callbacks=callbacks.get_list())
	
	print('\nCompleted VGG16 model training\n')
	
	return model


def data_generator(batch_size):
	
	while True:
		x_data = []
		y_data = []

		random_directories = np.random.choice(train_img_directories, batch_size)

		for directory in random_directories:
			dir_path = join(train_data_path, directory)
			img_name = np.random.choice(listdir(dir_path))

			x_data.append(preprocess_data(load_image(dir_path, img_name)))

			y_data.append(wnid_index_map[directory])

		yield np.array(x_data), to_categorical(y_data, 1000)


def load_image(img_path, img_name):
	img = image.load_img(join(img_path, img_name))
	return image.img_to_array(img)


def preprocess_data(img_array, target_size=(224, 224)):
	
	# change size of image
	img = image.array_to_img(img_array)
	height_width_tuple = (target_size[1], target_size[0])
	if img.size != height_width_tuple:
		img = img.resize(height_width_tuple)
		img_array = image.img_to_array(img)
	
	# change to BGR and subtract mean values
	if K.image_data_format() == 'channels_last':
		img_array = img_array[:, :, ::-1]
		img_array -= MEAN_VALUES.reshape((1, 1, 3))
	else:
		img_array = img_array[::-1, :, :]
		img_array -= MEAN_VALUES.reshape((3, 1, 1))
	
	return img_array


def postprocess_data(img_array):
	# add mean values
	img_array += MEAN_VALUES.reshape((1, 1, 3))
	
	# change back to RGB
	img_array = img_array[:, :, ::-1]

	return img_array


def main():
	# create a model, then train and test it.
	
	start_time = time()

	# model = create_model()
	model = create_model('imagenet', True)

	model = train(model)
	
	print('This took {:.2f} seconds'.format(time() - start_time))


main()
