from keras.engine import Model
from keras.layers import Flatten, Dense
from keras.optimizers import SGD
from keras_vggface.vggface import VGGFace
from keras.preprocessing.image import img_to_array, array_to_img
from keras.backend import image_data_format

import pickle
import math
import os
import random
import numpy as np
from PIL import Image

# import callbacks for visualizing
#from custom_keras.callbacks import CustomCallbacks

# find path to save networks and results
#save_path = os.path.dirname(__file__)

base_path = '/Users/annieaa/Documents/NTNU/Fordypningsprosjekt'

with open(os.path.join(base_path, 'imfdb_training_data.pickle'), 'rb') as f:
	training_data = pickle.load(f)


# custom parameters
nb_class = 98
hidden_dim = 512  # TODO: check if this is better than 1024

batch_size = 64
steps_per_epoch = math.ceil(len(training_data) / batch_size)
no_of_epochs = 5

img_size = (130, 130)

#MEAN_VALUES = np.array([112.9470, 83.4040, 72.5764])
MEAN_VALUES = np.array([93.5940, 104.7624, 129.1863]) # BGR from keras vgg-face github

if image_data_format() == 'channels_last':
	MEAN_VALUES = MEAN_VALUES.reshape(1, 1, 3)
else:
	MEAN_VALUES = MEAN_VALUES.reshape(3, 1, 1)


def create_model():

	vgg_model = VGGFace(include_top=True, input_shape=(130, 130, 3))

	for layer in vgg_model.layers:
		layer.trainable = False

	return vgg_model


def create_finetuning_model():

	vgg_model = VGGFace(include_top=False, input_shape=(130, 130, 3))

	for layer in vgg_model.layers:
		layer.trainable = False

	last_layer = vgg_model.output
	x = Flatten(name='flatten')(last_layer)
	x = Dense(hidden_dim, activation='relu', name='fc1')(x)
	x = Dense(hidden_dim, activation='relu', name='fc2')(x)
	out = Dense(nb_class, activation='softmax', name='predictions')(x)

	custom_vgg_model = Model(vgg_model.input, out)
	custom_vgg_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

	return custom_vgg_model


def train_model(model):

	# initialize custom callbacks
	'''callbacks = CustomCallbacks(save_path, preprocess_data, postprocess_data)
	callbacks.register_network_saver()
	callbacks.register_training_progress()
	callbacks.register_layer_activations()
	callbacks.register_saliency_maps()'''

	model.fit_generator(generator=data_generation(), steps_per_epoch=steps_per_epoch, epochs=no_of_epochs, verbose=1)
						#callbacks=callbacks.get_list())


def data_generation():

	while True:
		x_data = []
		y_data = []

		indices = random.sample(range(len(training_data)), batch_size)

		for i in indices:
			img_path, id_vector, expression_vector = training_data[i]
			img = Image.open(os.path.join(base_path, img_path))
			img_array = img_to_array(img)

			# alter to BGR
			if image_data_format() == 'channels_last':
				img_array = img_array[:, :, ::-1]
				img_array -= MEAN_VALUES.reshape((1, 1, 3))
			else:
				img_array = img_array[::-1, :, :]
				img_array -= MEAN_VALUES.reshape((3, 1, 1))

			x_data.append(img_array)
			y_data.append(id_vector)

		yield np.array(x_data), np.array(y_data)


def preprocess_data(img_array):

	# change size of and pad image if necessary
	image = array_to_img(img_array)
	width, height = image.size

	if width < height:
		new_width = math.ceil(img_size[1] * width / height)
		image = image.resize((new_width, img_size[1]), Image.ANTIALIAS)
	elif width > height:
		new_height = math.ceil(img_size[0] * height / width)
		image = image.resize((img_size[0], new_height), Image.ANTIALIAS)
	else:
		image = image.resize(img_size)

	new_image = Image.new('RGB', img_size)
	new_image.paste(image, (math.ceil((img_size[0] - image.size[0]) / 2),
							math.ceil((img_size[1] - image.size[1]) / 2)))

	img_array = img_to_array(new_image)

	# change to BGR and subtract mean values
	if image_data_format() == 'channels_last':
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
	model = create_finetuning_model()
	#train_model(model)


main()
