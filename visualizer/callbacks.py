from keras.callbacks import Callback
import keras.backend as K
from keras.preprocessing import image

import numpy as np
import pickle

from PIL import Image

from os import mkdir, listdir
from os.path import join, basename


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


class SaliencyMaps(Callback):

	def __init__(self, file_folder, interval=10):
		super(SaliencyMaps, self).__init__()
		self.results_folder = join(file_folder, 'results')
		self.interval = interval
		self.counter = 0

		# find image uploaded by user to use in visualization
		images_folder = join(file_folder, 'images')
		image_name = listdir(images_folder)[-1]
		image = Image.open(join(images_folder, image_name))

		# convert to correct format TODO: check if this is needed, and generalize
		self.input_tensor = np.array(image)[np.newaxis, :, :, np.newaxis]

	def on_batch_end(self, batch, logs={}):

		# only update visualization at user specified intervals
		if self.counter == self.interval:

			output_layer = self.model.layers[-1]
			# ignore the top layer of prediction if it is a softmax layer
			if output_layer.get_config()['activation'] == 'softmax':
				output_layer = self.model.layers[-2]

			predict_func = K.function([self.model.input, K.learning_phase()], [output_layer.output])
			# predict using the chosen image
			predictions = predict_func([self.input_tensor, 0])[0]

			# find the most likely predicted class
			max_class = np.argmax(predictions)

			# compute the gradient of the input with respect to the loss
			loss = output_layer.output[0, max_class]
			saliency = K.gradients(loss, self.model.input)[0]

			get_saliency_function = K.function([self.model.input, K.learning_phase()], [saliency])
			saliency = get_saliency_function([self.input_tensor, 0])[0][0][:, :, ::-1]

			# get the absolute value of the saliency
			abs_saliency = np.abs(saliency)

			# scale to fit between [0.0, 255.0]
			if abs_saliency.max() != 0.0:
				abs_saliency *= (255.0 / abs_saliency.max())

			file = join(self.results_folder, 'saliency_maps')
			np.save(file, abs_saliency)

			self.counter = 0

		self.counter += 1


class DeepVisualization(Callback):

	def __init__(self, file_folder, interval=1000):
		super(DeepVisualization, self).__init__()
		
		self.results_folder = join(file_folder, 'results')
		self.interval = interval
		self.counter = 0
		
		# find image uploaded by user to use in visualization
		images_folder = join(file_folder, 'images')
		img_name = listdir(images_folder)[-1]
		pil_img = Image.open(join(images_folder, img_name))
		
		# convert to array and add batch dimension
		self.img = np.expand_dims(image.img_to_array(pil_img), axis=0)
		
		# set channel dimension based on image data format from Keras backend
		if K.image_data_format() == 'channels_last':
			self.ch_dim = 3
		else:
			self.ch_dim = 1

	def on_batch_end(self, batch, logs={}):

		# only update visualization at user specified intervals
		if self.counter == self.interval:

			# perform deep visualization

			self.counter = 0

		self.counter += 1
