import pickle
import random
from math import ceil
from os.path import dirname, join

import numpy as np
from PIL import Image

from keras.engine import Model
from keras.layers import Input, Flatten, Dense, Concatenate
from keras_vggface.vggface import VGGFace
from keras.preprocessing.image import img_to_array, array_to_img
from keras.backend import image_data_format
from keras.optimizers import RMSprop

# import callbacks for visualizing
from custom_keras.callbacks import CustomCallbacks

# find path to save networks and results
save_path = dirname(__file__)

meta_path = '/home/mikaelbj/Documents/GitHub/training-visualizer/case_study/metadata'
data_path = '/home/mikaelbj/Documents/case_study_data'

# collect meta data files
with open(join(meta_path, 'IMFDB_training_meta.pickle'), 'rb') as f:
	training_meta = pickle.load(f)
with open(join(meta_path, 'IMFDB_validation_meta.pickle'), 'rb') as f:
	validation_meta = pickle.load(f)

# get emotion vector size
emotion_range = len(training_meta[0][2])

# custom parameters
experimental = False
nb_class = 98
hidden_dim = 512  # TODO: check if this is better than 1024

batch_size = 128
no_of_epochs = 10

img_size = (130, 130)

# compute steps for generators
steps_per_epoch = ceil(len(training_meta) / batch_size)
val_steps_per_epoch = ceil(len(validation_meta) / batch_size)

# MEAN_VALUES = np.array([112.9470, 83.4040, 72.5764])
MEAN_VALUES = np.array([93.5940, 104.7624, 129.1863])  # BGR mean values from keras vgg-face github

if image_data_format() == 'channels_last':
	MEAN_VALUES = MEAN_VALUES.reshape(1, 1, 3)
else:
	MEAN_VALUES = MEAN_VALUES.reshape(3, 1, 1)


def create_model():

	vgg_model = VGGFace(include_top=True, input_shape=(224, 224, 3))

	for layer in vgg_model.layers[:-1]:
		layer.trainable = False

	# get standard inputs from standard VGGFace model
	output_layer_input = vgg_model.layers[-1].input
	model_input = vgg_model.input

	if experimental:
		# define extra input
		expression_input = Input(shape=(emotion_range,))

		# add Concatenate layer to merge expression input with standard input
		concat = Concatenate()([vgg_model.layers[-1].input, expression_input])

		# update output layer input
		output_layer_input = concat

		# model should now receive two inputs
		model_input = [vgg_model.input, expression_input]

	output_layer = Dense(nb_class, activation='softmax', name='predictions')(output_layer_input)

	custom_vgg_model = Model(model_input, output_layer)
	custom_vgg_model.compile(optimizer=RMSprop(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

	return custom_vgg_model


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

	# initialize custom callbacks (visualization techniques will not work for experimental network)
	callbacks = CustomCallbacks(save_path, base_interval=20)
	callbacks.register_network_saver()
	callbacks.register_training_progress()

	model.fit_generator(generator=data_generation(training_meta), steps_per_epoch=steps_per_epoch, epochs=no_of_epochs,
						verbose=1, validation_data=data_generation(validation_meta), validation_steps=val_steps_per_epoch,
						callbacks=callbacks.get_list())


def data_generation(metadata):

	while True:
		images = []
		identifications = []
		expressions = []

		indices = random.sample(range(len(metadata)), batch_size)

		for i in indices:
			img_rel_path, id_vector, expression_vector = metadata[i]
			img = Image.open(join(data_path, img_rel_path))
			img = img.resize((224, 224))
			img_array = img_to_array(img)

			# alter to BGR and subtract mean values
			if image_data_format() == 'channels_last':
				img_array = img_array[:, :, ::-1]
				img_array -= MEAN_VALUES.reshape((1, 1, 3))
			else:
				img_array = img_array[::-1, :, :]
				img_array -= MEAN_VALUES.reshape((3, 1, 1))

			images.append(img_array)
			identifications.append(id_vector)
			expressions.append(expression_vector)

		data = np.array(images)
		targets = np.array(identifications)

		if experimental:
			data = [np.array(images), np.array(expressions)]

		yield data, targets


def preprocess_data(img_array):

	# change size of and pad image if necessary
	image = array_to_img(img_array)
	width, height = image.size

	if width < height:
		new_width = ceil(img_size[1] * width / height)
		image = image.resize((new_width, img_size[1]), Image.ANTIALIAS)
	elif width > height:
		new_height = ceil(img_size[0] * height / width)
		image = image.resize((img_size[0], new_height), Image.ANTIALIAS)
	else:
		image = image.resize(img_size)

	new_image = Image.new('RGB', img_size)
	new_image.paste(image, (ceil((img_size[0] - image.size[0]) / 2), ceil((img_size[1] - image.size[1]) / 2)))

	img_array = img_to_array(new_image)

	# change to BGR and subtract mean values
	if image_data_format() == 'channels_last':
		img_array = img_array[:, :, ::-1]
	else:
		img_array = img_array[::-1, :, :]

	img_array -= MEAN_VALUES

	return img_array


def postprocess_data(img_array):
	# add mean values
	img_array += MEAN_VALUES.reshape((1, 1, 3))

	# change back to RGB
	img_array = img_array[:, :, ::-1]

	return img_array


def main():

	# model = create_finetuning_model()
	model = create_model()
	train_model(model)


main()
