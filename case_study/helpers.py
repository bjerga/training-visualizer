import pickle
import random
from math import ceil
from os.path import join

import numpy as np
from PIL import Image

from keras.backend import image_data_format
from keras.models import load_model
from keras.preprocessing.image import img_to_array, array_to_img

# import callbacks for visualizing
from custom_keras.callbacks import CustomCallbacks

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

# collect meta data files for features
with open(join(case_path, 'metadata_feat', 'IMFDB_training_meta_feat.pickle'), 'rb') as f:
	training_meta_feat = pickle.load(f)
with open(join(case_path, 'metadata_feat', 'IMFDB_validation_meta_feat.pickle'), 'rb') as f:
	validation_meta_feat = pickle.load(f)
with open(join(case_path, 'metadata_feat', 'IMFDB_testing_meta_feat.pickle'), 'rb') as f:
	test_meta_feat = pickle.load(f)


# BGR mean values from GitHub repo of keras_vggface
MEAN_VALUES = np.array([93.5940, 104.7624, 129.1863])
MEAN_VALUES = MEAN_VALUES.reshape(1, 1, 3)


def train_model(model, no_of_epochs, batch_size, model_type, save_path):

	if model_type != 'baseline' and model_type != 'extra_input' and model_type != 'extra_output':
		raise ValueError('Unknown model type:', model_type)

	# initialize custom callbacks (visualization techniques will not work for experimental network)
	callbacks = CustomCallbacks(save_path)
	callbacks.register_network_saver()
	# training progress visualization does not work with extra output architectures
	if model_type != 'extra_output':
		callbacks.register_training_progress()

	# make generators and compute steps
	train_generator = feat_data_generator(training_meta_feat, batch_size, model_type, training=True)
	train_steps_per_epoch = ceil(len(training_meta) / batch_size)

	val_generator = feat_data_generator(validation_meta_feat, batch_size, model_type)
	val_steps_per_epoch = ceil(len(validation_meta) / batch_size)

	model.fit_generator(generator=train_generator, steps_per_epoch=train_steps_per_epoch, epochs=no_of_epochs, verbose=1,
						validation_data=val_generator, validation_steps=val_steps_per_epoch, callbacks=callbacks.get_list())


def test_model(model_path, batch_size, model_type, validation=False):

	model = load_model(model_path)

	if validation:
		generator = feat_data_generator(validation_meta_feat, batch_size, model_type)
		steps_per_epoch = ceil(len(validation_meta) / batch_size)
	else:
		generator = feat_data_generator(test_meta_feat, batch_size, model_type)
		steps_per_epoch = ceil(len(test_meta) / batch_size)

	metric_list = model.evaluate_generator(generator=generator, steps=steps_per_epoch)

	print('Test set results:')
	for i in range(len(model.metrics_names)):
		print(model.metrics_names[i], metric_list[i])


def feat_data_generator(metadata, batch_size, model_type, training=False):

	start = 0
	stop = batch_size

	while True:
		features = []
		identifications = []
		expressions = []

		if training:
			# if training, randomly sample
			indices = random.sample(range(len(metadata)), batch_size)
		else:
			# if not, iterate over all data

			if start > len(metadata):
				# reset
				start = 0
				stop = batch_size
			elif stop > len(metadata):
				stop = len(metadata)

			indices = range(start, stop)

			start += batch_size
			stop += batch_size

		for i in indices:
			feature_vector, id_vector, expression_vector = metadata[i]

			features.append(feature_vector)
			identifications.append(id_vector)
			expressions.append(expression_vector)

		data = np.array(features)
		targets = np.array(identifications)

		if model_type == 'extra_input':
			data = [data, np.array(expressions)]
		elif model_type == 'extra_output':
			targets = [targets, np.array(expressions)]

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
