from keras.callbacks import Callback
import keras.backend as K

import numpy as np
import pickle

from os import mkdir
from os.path import join, basename


# saves the network at correct path when starting training and after each epoch
class NetworkSaver(Callback):

	def __init__(self, save_path):
		super(NetworkSaver, self).__init__()
		self.networks_path = join(save_path, 'networks')
		self.name = basename(save_path)

	def on_train_begin(self, logs={}):
		# make path, if not exists
		try:
			mkdir(self.networks_path)
			print('networks-folder created')
		except FileExistsError:
			# file exists, which is what we want
			pass

		self.model.save(join(self.networks_path, self.name + '.h5'))

	def on_epoch_end(self, batch, logs={}):
		self.model.save(join(self.networks_path, self.name + '.h5'))


# saves accuracy at each finished training batch
class AccuracyListSaver(Callback):

	def __init__(self, save_path):
		super(AccuracyListSaver, self).__init__()
		self.results_path = join(save_path, 'results')

	def on_train_begin(self, logs={}):
		# ensure file creation
		with open(join(self.results_path, 'batch_accuracy.txt'), 'w') as f:
			f.write('')

	def on_batch_end(self, batch, logs={}):
		# write new accuracy line
		with open(join(self.results_path, 'batch_accuracy.txt'), 'a') as f:
			f.write(str(logs['acc']) + '\n')  # saves loss at each finished training batch


class LossListSaver(Callback):

	def __init__(self, save_path):
		super(LossListSaver, self).__init__()
		self.results_path = join(save_path, 'results')

	def on_train_begin(self, logs={}):
		# ensure file creation
		with open(join(self.results_path, 'batch_loss.txt'), 'w') as f:
			f.write('')

	def on_batch_end(self, batch, logs={}):
		# write new loss line
		with open(join(self.results_path, 'batch_loss.txt'), 'a') as f:
			f.write(str(logs['loss']) + '\n')


# saves activation arrays for each layer as tuples: (layer-name, array)
class ActivationTupleListSaver(Callback):

	input_tensor = None

	def __init__(self, save_path):

		super(ActivationTupleListSaver, self).__init__()
		self.results_path = join(save_path, 'results')

		# get one random image from training data to use as input
		training_data, _, _, _ = load_data()
		self.input_tensor = training_data[np.random.randint(len(training_data))].reshape(1, 28, 28, 1)

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

		with open(join(self.results_path, 'layer_activations.pickle'), 'wb') as f:
			pickle.dump(layer_tuples, f)


# TODO: this is specific for mnist, and should therefore not be in this script

from keras.datasets import mnist
from keras.utils.np_utils import to_categorical


def load_data():
	# load
	(training_data, training_targets), (test_data, test_targets) = mnist.load_data()

	# reshape data
	training_data = training_data.reshape(training_data.shape[0], 28, 28, 1)
	test_data = test_data.reshape(test_data.shape[0], 28, 28, 1)

	# normalize data
	training_data = training_data.astype('float32')
	test_data = test_data.astype('float32')
	training_data /= 255.0
	test_data /= 255.0

	# make targets categorical
	training_targets = to_categorical(training_targets, 10)
	test_targets = to_categorical(test_targets, 10)

	return training_data, training_targets, test_data, test_targets
