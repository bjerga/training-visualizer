from keras.callbacks import Callback
import keras.backend as K

import numpy as np
import pickle

from PIL import Image

from os import mkdir, listdir
from os.path import join, basename


# saves the network at correct path when starting training and after each epoch
class NetworkSaver(Callback):

	def __init__(self, file_folder):
		super(NetworkSaver, self).__init__()
		self.networks_folder = join(file_folder, 'networks')
		self.name = basename(file_folder)

	def on_train_begin(self, logs={}):
		# make path, if not exists
		try:
			mkdir(self.networks_folder)
			print('networks-folder created')
		except FileExistsError:
			# file exists, which is what we want
			pass

		self.model.save(join(self.networks_folder, self.name + '.h5'))

	def on_epoch_end(self, batch, logs={}):
		self.model.save(join(self.networks_folder, self.name + '.h5'))


# saves accuracy at each finished training batch
class AccuracyListSaver(Callback):

	def __init__(self, file_folder):
		super(AccuracyListSaver, self).__init__()
		self.results_folder = join(file_folder, 'results')

	def on_train_begin(self, logs={}):
		# ensure file creation
		with open(join(self.results_folder, 'batch_accuracy.txt'), 'w') as f:
			f.write('')

	def on_batch_end(self, batch, logs={}):
		# write new accuracy line
		with open(join(self.results_folder, 'batch_accuracy.txt'), 'a') as f:
			f.write(str(logs['acc']) + '\n')  # saves loss at each finished training batch


class LossListSaver(Callback):

	def __init__(self, file_folder):
		super(LossListSaver, self).__init__()
		self.results_folder = join(file_folder, 'results')

	def on_train_begin(self, logs={}):
		# ensure file creation
		with open(join(self.results_folder, 'batch_loss.txt'), 'w') as f:
			f.write('')

	def on_batch_end(self, batch, logs={}):
		# write new loss line
		with open(join(self.results_folder, 'batch_loss.txt'), 'a') as f:
			f.write(str(logs['loss']) + '\n')


# saves activation arrays for each layer as tuples: (layer-name, array)
class ActivationTupleListSaver(Callback):

	input_tensor = None

	def __init__(self, file_folder):

		super(ActivationTupleListSaver, self).__init__()
		self.results_folder = join(file_folder, 'results')

		# TODO: make sure that it is OK not to handle whether the image folder is empty
		# get visualization image corresponding to the file
		images_folder = join(file_folder, 'images')
		image_name = listdir(images_folder)[-1]
		image = Image.open(join(images_folder, image_name))

		# set input tensor and reshape to (1, width, height, 1)
		self.input_tensor = np.array(image)[np.newaxis, :, :, np.newaxis]

	def on_epoch_end(self, batch, logs={}):

		# initialize layer tuple list with image
		layer_tuples = []

		# for all layers, get and save activation tensor
		for layer in self.model.layers:
			# create function using keras-backend for getting activation tensor
			get_activation_tensor = K.function([self.model.input, K.learning_phase()], [layer.output])

			# save tuple (layer name, layer's activation tensor)
			# NOTE: learning phase 0 is testing and 1 is training (difference unknown as this point)
			layer_tuples.append((layer.name, get_activation_tensor([self.input_tensor, 0])[0]))

		with open(join(self.results_folder, 'layer_activations.pickle'), 'wb') as f:
			pickle.dump(layer_tuples, f)