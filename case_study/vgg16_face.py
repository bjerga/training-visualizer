import pickle
import random
from math import ceil
from os.path import dirname, join

import numpy as np
from PIL import Image

from keras.models import Model
from keras.layers import Input, Dense, Concatenate, Dropout
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
with open(join(case_path, 'metadata', 'IMFDB_testing_meta.pickle'), 'rb') as f:
	test_meta = pickle.load(f)

# TODO: use flipped for now
# collect meta data files for features
with open(join(case_path, 'metadata_feat', 'IMFDB_training_meta_feat.pickle'), 'rb') as f:
	training_meta_feat = pickle.load(f)
with open(join(case_path, 'metadata_feat', 'IMFDB_validation_meta_feat.pickle'), 'rb') as f:
	validation_meta_feat = pickle.load(f)
with open(join(case_path, 'metadata_feat', 'IMFDB_testing_meta_feat.pickle'), 'rb') as f:
	test_meta_feat = pickle.load(f)


# set model parameters
# choose if model is to be experimental (False & False = baseline)
extra_input = False
extra_output = True
assert not (extra_input and extra_output)
# amount of classification possibilities
id_amount = 98

# set training parameters
no_of_epochs = 100
batch_size = 512

# get expression vector size
expression_amount = len(training_meta[0][2])

# compute steps for generators
steps_per_epoch = ceil(len(training_meta) / batch_size)
val_steps_per_epoch = ceil(len(validation_meta) / batch_size)

# BGR mean values from GitHub repo of keras_vggface
MEAN_VALUES = np.array([93.5940, 104.7624, 129.1863])

if image_data_format() == 'channels_last':
	MEAN_VALUES = MEAN_VALUES.reshape(1, 1, 3)
	input_shape = (224, 224, 3)
else:
	MEAN_VALUES = MEAN_VALUES.reshape(3, 1, 1)
	input_shape = (3, 224, 224)


"""

IMPORTANT NOTE:

In VGGFace, the top layers have activations in its own layers, therefore input and output tensors of last layer
should be accessed in the following manner (see numbers):

input_tensor = model.layers[-2].input
output_tensor = model.layers[-1].output

"""


# create model only consisting of top layers using features from a VGGFace base model
def create_top_model():

	# load VGGFace model
	vgg_face_model = VGGFace(include_top=True, input_shape=input_shape)

	# get output shape of VGGFace base model (excluding last layer, see note)
	base_output_shape = vgg_face_model.layers[-2].input_shape[1:]

	# define model input
	model_input = Input(shape=base_output_shape, name='feat_input')

	if extra_input:
		# define extra input
		expression_input = Input(shape=(expression_amount,), name='expression_input')

		# define experimental structure
		# use Concatenate layer to merge expression input with standard input
		x = model_input
		x = Dropout(0.5)(x)
		x = Dense(1024, activation='relu', name='fc_exp1')(x)
		x = Dropout(0.5)(x)
		x = Dense(1024, activation='relu', name='fc_exp2')(x)
		x = Dropout(0.5)(x)
		x = Concatenate()([x, expression_input])
		x = Dense(1024, activation='relu', name='fc_exp3')(x)
		x = Dropout(0.5)(x)
		x = Dense(1024, activation='relu', name='fc_exp4')(x)
		x = Dropout(0.5)(x)
		id_output = Dense(id_amount, activation='softmax', name='id_output')(x)
		model_output = id_output

		# redefine model input to receive two inputs
		model_input = [model_input, expression_input]

	elif extra_output:

		# define experimental structure
		x = model_input
		x = Dropout(0.5)(x)
		x = Dense(1024, activation='relu', name='fc_exp')(x)
		x = Dropout(0.5)(x)
		x = Dense(1024, activation='relu', name='fc_exp1')(x)
		x_id = Dropout(0.5)(x)
		x = Dense(1024, activation='relu', name='fc_exp2')(x_id)
		x = Dropout(0.5)(x)
		id_output = Dense(id_amount, activation='softmax', name='id_output')(x)

		# define extra output (with extra layers)
		x = x_id
		x = Dense(1024, activation='relu', name='fc_exp3')(x)
		x = Dropout(0.5)(x)
		# x = Dense(1024, activation='relu', name='fc_exp3')(x)
		# x = Dropout(0.5)(x)
		# x = Dense(512, activation='relu', name='fc_exp4')(x)
		# x = Dropout(0.5)(x)
		expression_output = Dense(expression_amount, activation='softmax', name='expression_output')(x)

		# define model output to yield two outputs
		# model_output = [id_output, expression_output]
		model_output = Concatenate()([id_output, expression_output])

	else:
		# define baseline structure
		x = model_input
		x = Dropout(0.5)(x)
		x = Dense(1024, activation='relu', name='fc_exp')(x)
		x = Dropout(0.5)(x)
		x = Dense(1024, activation='relu', name='fc_exp1')(x)
		x = Dropout(0.5)(x)
		x = Dense(1024, activation='relu', name='fc_exp2')(x)
		x = Dropout(0.5)(x)
		id_output = Dense(id_amount, activation='softmax', name='id_output')(x)
		model_output = id_output

	# create and compile model
	custom_vgg_model = Model(model_input, model_output)
	custom_vgg_model.compile(optimizer=Adam(lr=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])
	# custom_vgg_model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

	return custom_vgg_model


def train_model(model):

	# initialize custom callbacks (visualization techniques will not work for experimental network)
	callbacks = CustomCallbacks(save_path)
	callbacks.register_network_saver()
	callbacks.register_training_progress()

	train_generator = feat_data_generator(training_meta_feat)
	val_generator = feat_data_generator(validation_meta_feat)

	model.fit_generator(generator=train_generator, steps_per_epoch=steps_per_epoch, epochs=no_of_epochs, verbose=1,
						validation_data=val_generator, validation_steps=val_steps_per_epoch, callbacks=callbacks.get_list())


def test_model(model):
	


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

		if extra_input:
			data = [data, np.array(expressions)]
		elif extra_output:
			targets = np.array([[*id_vector, *expression_vector] for id_vector, expression_vector in zip(identifications, expressions)])

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

	model = create_top_model()
	train_model(model)


main()
