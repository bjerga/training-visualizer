import pickle
import random
from math import ceil
from os.path import dirname, join

import numpy as np
from PIL import Image

from keras.models import Model
from keras.layers import Input, Dense, Concatenate
from keras_vggface.vggface import VGGFace
from keras.preprocessing.image import img_to_array, array_to_img
from keras.backend import image_data_format
from keras.optimizers import Adam

# import callbacks for visualizing
from custom_keras.callbacks import CustomCallbacks

# find path to save networks and results
save_path = dirname(__file__)

# set paths to case study file and data
case_path = '/home/mikaelbj/Documents/GitHub/training-visualizer/case_study'
data_path = '/home/mikaelbj/Documents/case_study_data'

# collect meta data files for images
with open(join(case_path, 'metadata', 'IMFDB_training_meta.pickle'), 'rb') as f:
	training_meta = pickle.load(f)
with open(join(case_path, 'metadata', 'IMFDB_validation_meta.pickle'), 'rb') as f:
	validation_meta = pickle.load(f)

# collect meta data files for features
with open(join(case_path, 'metadata_feat', 'IMFDB_training_meta_feat.pickle'), 'rb') as f:
	training_meta_feat = pickle.load(f)
with open(join(case_path, 'metadata_feat', 'IMFDB_validation_meta_feat.pickle'), 'rb') as f:
	validation_meta_feat = pickle.load(f)


# set model parameters
# choose if model is to be experimental
experimental = False
# choose if model should only consist of top layers (if true, input is features instead of images)
only_top = True
# amount of classification possibilities
class_amount = 98

# set training parameters
no_of_epochs = 10
batch_size = 128

# get emotion vector size
emotion_range = len(training_meta[0][2])

# compute steps for generators
steps_per_epoch = ceil(len(training_meta) / batch_size)
val_steps_per_epoch = ceil(len(validation_meta) / batch_size)

# BGR mean values from GitHub repo of keras_vggface
MEAN_VALUES = np.array([93.5940, 104.7624, 129.1863])

if image_data_format() == 'channels_last':
	MEAN_VALUES = MEAN_VALUES.reshape(1, 1, 3)
else:
	MEAN_VALUES = MEAN_VALUES.reshape(3, 1, 1)

"""

IMPORTANT NOTE:

In VGGFace, the top layers have activations in its own layers, therefore input and output tensors of last layer
should be accessed in the following manner (see numbers):

input_tensor = model.layers[-2].input
output_tensor = model.layers[-1].output

"""


def create_model():

	vgg_model = VGGFace(include_top=True, input_shape=(224, 224, 3))

	for layer in vgg_model.layers[:-1]:
		layer.trainable = False

	# get standard inputs from standard VGGFace model
	model_input = vgg_model.input
	output_layer_input = vgg_model.layers[-2].input

	if only_top:
		model_input = Input(shape=vgg_model.layers[-2].input_shape[1:], name='feat_input')
		output_layer_input = model_input

	if experimental:
		# define extra input
		expression_input = Input(shape=(emotion_range,), name='emo_input')

		# model should now receive two inputs
		model_input = [model_input, expression_input]

		# update output layer input to be Concatenate layer (merge expression input with standard input)
		output_layer_input = Concatenate()([output_layer_input, expression_input])

	output_layer = Dense(class_amount, activation='softmax', name='predictions')(output_layer_input)

	custom_vgg_model = Model(model_input, output_layer)
	custom_vgg_model.compile(optimizer=Adam(lr=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])

	return custom_vgg_model


def train_model(model):

	# initialize custom callbacks (visualization techniques will not work for experimental network)
	callbacks = CustomCallbacks(save_path)
	callbacks.register_network_saver()
	callbacks.register_training_progress()

	train_generator = image_data_generator(training_meta)
	val_generator = image_data_generator(validation_meta)

	if only_top:
		train_generator = feat_data_generator(training_meta_feat)
		val_generator = feat_data_generator(validation_meta_feat)

	model.fit_generator(generator=train_generator, steps_per_epoch=steps_per_epoch, epochs=no_of_epochs, verbose=1,
						validation_data=val_generator, validation_steps=val_steps_per_epoch, callbacks=callbacks.get_list())


def image_data_generator(metadata):

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
			data = [data, np.array(expressions)]

		yield data, targets


def feat_data_generator(metadata):

	while True:
		features = []
		identifications = []
		expressions = []

		indices = random.sample(range(len(metadata)), batch_size)

		for i in indices:
			feature_vector, id_vector, expression_vector = metadata[i]

			features.append(feature_vector)
			identifications.append(id_vector)
			expressions.append(expression_vector)

		data = np.array(features)
		targets = np.array(identifications)

		if experimental:
			data = [data, np.array(expressions)]

		yield data, targets


def preprocess_data(img_array):

	# set image size
	img_size = (130, 130)

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
