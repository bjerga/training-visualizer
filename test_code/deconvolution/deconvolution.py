import numpy as np

from time import time
from os.path import dirname, join

import keras.backend as K
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image

from visualizer.custom_keras_models import DeconvolutionModel

# VGG16 mean values
MEAN_VALUES = np.array([103.939, 116.779, 123.68])

# define output path and make folder
output_path = join(dirname(__file__), 'deconv_output')


def load_image_from_file(img_name):
	img_path = join(dirname(__file__), 'deconv_input', img_name)
	img = image.load_img(img_path, target_size=(224, 224))
	img = image.img_to_array(img)
	
	if K.image_data_format() == 'channels_last':
		img -= MEAN_VALUES.reshape((1, 1, 3))
	else:
		img -= MEAN_VALUES.reshape((3, 1, 1))

	return np.expand_dims(img, axis=0)


def deconv_example():
	img_name = 'dog.jpg'
	img = load_image_from_file(img_name)
	
	conv_model = VGG16(include_top=False, weights='imagenet', input_shape=img.shape[1:])
	
	# print conv info
	# print('\n***CONVOLUTIONAL MODEL INFO***')
	# print('Conv. input shape:', conv_model.input_shape)
	# print('Conv. output shape:', conv_model.output_shape)
	# print('\nLayers in conv. model:')
	# for layer in conv_model.layers:
	# 	print(layer.name)
	
	print('\nCreating deconvolution model')
	start_time = time()
	deconv_model = DeconvolutionModel(conv_model, img, img_name, output_path)
	print('\nTime to create was {:.4f} seconds'.format(time() - start_time))
	
	# note that layers are zero indexed
	feat_map_layer_no = 18
	
	choose_max_images = False
	
	print('\nReady for deconv. pred.')
	start_time = time()
	
	if choose_max_images:
		deconv_model.produce_reconstruction_from_top_images(feat_map_layer_no, 100, 5, 3)
		# deconv_model.produce_reconstruction_from_top_images(feat_map_layer_no, 100, 5, feat_map_nos=[88, 351, 178])
	else:
		deconv_model.produce_reconstruction_with_fixed_image(feat_map_layer_no, 10)
		# deconv_model.produce_reconstruction_with_fixed_image(feat_map_layer_no, feat_map_nos=[88, 351, 178, 0, 5])
	
	print('\nTime to perform reconstructions for feat maps was {:.4f} seconds'.format(time() - start_time))


deconv_example()
