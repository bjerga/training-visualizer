import pickle
from math import ceil
from time import time
from os import listdir
from os.path import join, dirname

import numpy as np

import keras.backend as K
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Conv2D, MaxPooling2D
from keras.optimizers import RMSprop
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.utils.np_utils import to_categorical

# import callbacks for visualizing
from custom_keras.callbacks import CustomCallbacks

# find path to save networks and results
save_path = dirname(__file__)

# find path to imagenet URLs
imagenet_path = '/media/mikaelbj/Mikal My Book/ImageNet'
train_data_path = join(imagenet_path, 'ILSVRC2012_img_train')
test_data_path = join(imagenet_path, 'ILSVRC2012_img_test')
val_data_path = join(imagenet_path, 'ILSVRC2012_img_val')

train_img_directories = listdir(train_data_path)
train_data_amount = 1281167

MEAN_VALUES = np.array([103.939, 116.779, 123.68])

if K.image_data_format() == 'channels_last':
	input_shape = (224, 224, 3)
else:
	input_shape = (3, 224, 224)

with open(join(imagenet_path, 'wnid_index_map.pickle'), 'rb') as f:
	wnid_index_map = pickle.load(f)


def create_model():
	# define model
	model = VGG16(include_top=True, weights=None, input_shape=input_shape)
	model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

	print('\nModel successfully created')
	
	return model


def create_model_untrainable():

	model = VGG16(include_top=True, weights='imagenet', input_shape=input_shape)
	for i in range(len(model.layers)):
		model.layers[i].trainable = False
	model.compile(optimizer=RMSprop(lr=0), loss='categorical_crossentropy', metrics=['accuracy'])

	print('\nModel successfully created')
	
	return model


def create_model_new_top():

	img_input = Input(shape=input_shape)

	# Block 1
	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

	# Block 2
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

	# Block 3
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

	# Block 4
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

	# Block 5
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

	# Classification block
	x = Flatten(name='flatten')(x)
	x = Dense(4096, activation='relu', name='fc1')(x)
	x = Dense(4096, activation='relu', name='fc2')(x)
	x = Dense(1000, activation='softmax', name='predictions')(x)

	base_model = VGG16(include_top=True, weights='imagenet', input_shape=input_shape)

	model = Model(inputs=img_input, outputs=x)

	for i in range(len(base_model.layers[:-4])):
		model.layers[i].set_weights(base_model.layers[i].get_weights())
		model.layers[i].trainable = False

	model.compile(optimizer=RMSprop(lr=0.000001), loss='categorical_crossentropy', metrics=['accuracy'])

	print('\nModel successfully created')

	return model


def train(model, no_of_epochs=50):
	# train top layers of model (self-defined layers)
	print('\n\nCommence VGG16 model training\n')

	batch_size = 64
	steps_per_epoch = ceil(train_data_amount / batch_size)
	
	# initialize custom callbacks
	callbacks = CustomCallbacks(save_path, preprocess_data, postprocess_data)
	callbacks.register_network_saver()
	callbacks.register_training_progress()
	callbacks.register_layer_activations()
	callbacks.register_saliency_maps()
	callbacks.register_deconvolution_network(18, 10, interval=10)
	# callbacks.register_deep_visualization([(-1, 402), (-1, 587), (-1, 950)], 2500.0, 500, l2_decay=0.0001,
	# 									  blur_interval=4, blur_std=1.0, interval=10)

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
	height_weight_tuple = (target_size[1], target_size[0])
	if img.size != height_weight_tuple:
		img = img.resize(height_weight_tuple)
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
	# change back to RGB
	img_array = img_array[:, :, ::-1]

	# subtract mean values
	img_array += MEAN_VALUES.reshape((1, 1, 3))
	return img_array


def main():
	# create a model, then train and test it.
	
	start_time = time()

	# model = create_model()
	model = create_model_untrainable()
	# model = create_model_new_top()

	model = train(model)
	
	print('This took %.2f seconds' % (time() - start_time))


main()
