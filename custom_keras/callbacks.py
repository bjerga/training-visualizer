import numpy as np
import pickle

from math import ceil
from os import mkdir, listdir
from os.path import join, basename

from scipy.misc import toimage
from scipy.ndimage.filters import gaussian_filter
from PIL import Image

import keras.backend as K
from keras.models import Model
from keras.layers import InputLayer, Dropout, Flatten
from keras.preprocessing import image
from keras.callbacks import Callback

from custom_keras.models import DeconvolutionModel

import datetime
from shutil import copytree


# choose which layers to exclude from layer activation visualization by default
EXCLUDE_LAYERS = (InputLayer, Dropout, Flatten)


class BackupResults(Callback):

	def __init__(self, file_folder, backup_folder, interval):
		super(BackupResults, self).__init__()

		self.results_folder = join(file_folder, 'results')
		self.backup_folder = backup_folder
		self.interval = interval
		self.counter = 0

	def on_train_begin(self, logs=None):
		try:
			mkdir(self.backup_folder)
		except FileExistsError:
			# file exists, which is what we want
			pass

	def on_batch_end(self, epoch, logs=None):

		self.counter += 1

		if self.counter == self.interval:

			timestamp = "{:%Y-%m-%d %H:%M:%S}".format(datetime.datetime.now())
			copytree(self.results_folder, join(self.backup_folder, timestamp))

			self.counter = 0


class CustomCallbacks:

	def __init__(self, file_folder, custom_preprocess=None, custom_postprocess=None, base_interval=10):
		
		self.file_folder = file_folder
		self.custom_preprocess = custom_preprocess
		self.custom_postprocess = custom_postprocess
		self.base_interval = base_interval
		
		# add list to store all callbacks registered in
		self.callback_list = []
		
	def get_list(self):
		return self.callback_list

	def register_backup_results(self, backup_folder, interval):
		self.callback_list.append(BackupResults(self.file_folder, backup_folder, interval))
		
	def register_network_saver(self):
		self.callback_list.append(NetworkSaver(self.file_folder))
		
	def register_training_progress(self):
		self.callback_list.append(TrainingProgress(self.file_folder))
		
	def register_layer_activations(self, exclude_layers=EXCLUDE_LAYERS, interval=None):
		if interval is None:
			interval = self.base_interval
		self.callback_list.append(LayerActivations(self.file_folder, exclude_layers, self.custom_preprocess, interval))
		
	def register_saliency_maps(self, interval=None):
		if interval is None:
			interval = self.base_interval
		self.callback_list.append(SaliencyMaps(self.file_folder, self.custom_preprocess, self.custom_postprocess, interval))
		
	def register_deconvolution_network(self, feat_map_layer_no, feat_map_amount=None, feat_map_nos=None,
									   custom_keras_model_info=None, interval=None):
		if interval is None:
			interval = self.base_interval
		self.callback_list.append(DeconvolutionNetwork(self.file_folder, feat_map_layer_no, feat_map_amount, feat_map_nos,
													   self.custom_preprocess, self.custom_postprocess, custom_keras_model_info, interval))
		
	def register_deep_visualization(self, units_to_visualize, learning_rate, no_of_iterations, l2_decay=0, blur_interval=0,
									blur_std=0, value_percentile=0, norm_percentile=0, contribution_percentile=0,
									abs_contribution_percentile=0, interval=None):
		if interval is None:
			interval = self.base_interval
		self.callback_list.append(DeepVisualization(self.file_folder, units_to_visualize, learning_rate, no_of_iterations,
													l2_decay, blur_interval, blur_std, value_percentile, norm_percentile,
													contribution_percentile, abs_contribution_percentile, self.custom_postprocess, interval))


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


class TrainingProgress(Callback):

	def __init__(self, file_folder):
		super(TrainingProgress, self).__init__()
		self.results_folder = join(file_folder, 'results')
		self.batches_in_epoch = None
		self.epoch = 0

	def on_train_begin(self, logs={}):
		try:
			# assume regular fit-method is being used
			self.batches_in_epoch = ceil(self.params['samples'] / self.params['batch_size'])
		except KeyError:
			# params are missing 'batch_size' key, fit_generator is being used
			self.batches_in_epoch = self.params['steps']

		# ensure file creation
		with open(join(self.results_folder, 'training_progress.txt'), 'w') as f:
			f.write('')
		if self.params['do_validation']:
			with open(join(self.results_folder, 'training_progress_val.txt'), 'w') as f:
				f.write('')

	def on_batch_end(self, batch, logs={}):
		# write new accuracy line
		with open(join(self.results_folder, 'training_progress.txt'), 'a') as f:
			# saves accuracy at each finished training batch as lines of "x-value acc loss"
			f.write("{} {} {}\n".format(self.epoch + (batch / self.batches_in_epoch), logs['acc'], logs['loss']))
		
	def on_epoch_end(self, epoch, logs={}):
		self.epoch += 1
		if self.params['do_validation']:
			with open(join(self.results_folder, 'training_progress_val.txt'), 'a') as f:
				# saves validation accuracy at each finished training epoch
				f.write("{} {:.5f} {:.5f}\n".format(epoch + 1, logs['val_acc'], logs['val_loss']))


# saves activation arrays for each layer as tuples: (layer-name, array)
class LayerActivations(Callback):

	def __init__(self, file_folder, exclude_layers=EXCLUDE_LAYERS, custom_preprocess=None, interval=10):

		super(LayerActivations, self).__init__()
		self.results_folder = join(file_folder, 'results')
		self.interval = interval
		self.counter = 0

		self.exclude_layers = exclude_layers

		# find image uploaded by user to use in visualization
		images_folder = join(file_folder, 'images')
		img_name = listdir(images_folder)[-1]
		
		# load image as array
		self.img_array = image.img_to_array(Image.open(join(images_folder, img_name)))
		
		# if supplied, apply custom preprocessing
		if custom_preprocess is not None:
			self.img_array = custom_preprocess(self.img_array)
		
		# add batch dimension
		self.img_array = np.expand_dims(self.img_array, 0)

	def on_batch_end(self, batch, logs={}):

		self.counter += 1

		# initialize layer tuple list with image
		layer_tuples = []

		# only update visualization at user specified intervals
		if self.counter == self.interval:

			for layer_no in range(len(self.model.layers)):

				layer = self.model.layers[layer_no]

				# check if layer should be included
				if not isinstance(layer, tuple(self.exclude_layers)):

					# create function using keras-backend for getting activation array
					get_activation_array = K.function([self.model.input, K.learning_phase()], [layer.output])
					# use function to find activation array for the chosen image
					act_array = get_activation_array([self.img_array, 0])[0]
					
					# remove batch dimension
					act_array = act_array[0]
					
					# if theano dimensions
					if K.image_data_format() == 'channels_first':
						# if greyscale image with no color dimension, add dimension
						if len(act_array.shape) == 2:
							act_array = np.expand_dims(act_array, 0)
						# alter dimensions from (color, height, width) to (height, width, color)
						if len(act_array.shape) == 3:
							act_array = act_array.transpose((1, 2, 0))

					# scale to fit between [0.0, 255.0]
					if act_array.max() != 0.0:
						act_array *= (255.0 / act_array.max())

					if len(act_array.shape) == 3:
						# get on correct format (list of filters)
						act_array = np.rollaxis(act_array, 2)

					# save tuple (layer name, layer's activation tensor)
					layer_tuples.append(("Layer {0}: {1}".format(layer_no, layer.name), act_array))

			with open(join(self.results_folder, 'layer_activations.pickle'), 'wb') as f:
				pickle.dump(layer_tuples, f)

			self.counter = 0


class SaliencyMaps(Callback):

	def __init__(self, file_folder, custom_preprocess=None, custom_postprocess=None, interval=10):
		super(SaliencyMaps, self).__init__()
		self.results_folder = join(file_folder, 'results')
		self.interval = interval
		self.counter = 0
		
		self.custom_preprocess = custom_preprocess
		self.custom_postprocess = custom_postprocess

		# find image uploaded by user to use in visualization
		images_folder = join(file_folder, 'images')
		img_name = listdir(images_folder)[-1]
		
		# load image as array
		self.img_array = image.img_to_array(Image.open(join(images_folder, img_name)))
		
		# if supplied, apply custom preprocessing
		if self.custom_preprocess is not None:
			self.img_array = self.custom_preprocess(self.img_array)
		
		# add batch dimension
		self.img_array = np.expand_dims(self.img_array, 0)
		
	def on_train_begin(self, logs=None):

		# set which output tensor of model to use
		self.output_tensor = self.model.output
		# if last layer uses softmax activation
		if self.model.layers[-1].get_config()['activation'] == 'softmax':
			# update chosen output tensor to be output tensor of previous layer
			self.output_tensor = self.model.layers[-1].input

		# set prediction function based on output tensor chosen
		self.predict_func = K.function([self.model.input, K.learning_phase()], [self.output_tensor])

	def on_batch_end(self, batch, logs={}):

		self.counter += 1

		# only update visualization at user specified intervals
		if self.counter == self.interval:

			# predict using the chosen image
			predictions = self.predict_func([self.img_array, 0])[0]

			# find the most likely predicted class
			max_class = np.argmax(predictions)

			# compute the gradients of loss w.r.t. the input image
			loss = self.output_tensor[0, max_class]
			saliency = K.gradients(loss, self.model.input)[0]

			get_saliency_function = K.function([self.model.input, K.learning_phase()], [saliency])
			saliency = get_saliency_function([self.img_array, 0])[0]
			
			# remove batch dimension
			saliency = saliency[0]
			
			# if theano dimensions
			if K.image_data_format() == 'channels_first':
				# if greyscale image with no color dimension, add dimension
				if len(saliency.shape) == 2:
					saliency = np.expand_dims(saliency, 0)
				# alter dimensions from (color, height, width) to (height, width, color)
				saliency = saliency.transpose((1, 2, 0))

			# get the absolute value of the saliency
			abs_saliency = np.abs(saliency)

			# convert from RGB to greyscale (take max of each RGB value)
			abs_saliency = np.amax(abs_saliency, axis=2)

			# add inner channel dimension
			abs_saliency = np.expand_dims(abs_saliency, axis=3)

			# scale to fit between [0.0, 255.0]
			if abs_saliency.max() != 0.0:
				abs_saliency *= (255.0 / abs_saliency.max())

			# convert to uint8
			abs_saliency = abs_saliency.astype('uint8')

			with open(join(self.results_folder, 'saliency_maps.pickle'), 'wb') as f:
				pickle.dump(abs_saliency, f)

			self.counter = 0


class DeconvolutionNetwork(Callback):
	def __init__(self, file_folder, feat_map_layer_no, feat_map_amount=None, feat_map_nos=None, custom_preprocess=None,
				 custom_postprocess=None, custom_keras_model_info=None, interval=100):
		super(DeconvolutionNetwork, self).__init__()
		
		self.results_folder = join(file_folder, 'results')
		self.interval = interval
		self.counter = 0
		
		# find image uploaded by user to use in visualization
		images_folder = join(file_folder, 'images')
		img_name = listdir(images_folder)[-1]
		
		# load image as array
		self.img_array = image.img_to_array(Image.open(join(images_folder, img_name)))
		
		# used for reconstruction production
		self.feat_map_layer_no = feat_map_layer_no
		self.feat_map_amount = feat_map_amount
		self.feat_map_nos = feat_map_nos
		
		# deconvolution model info
		self.deconv_model = None
		self.custom_keras_model_info = custom_keras_model_info
		
		# save pre- and postprocessing methods
		self.custom_preprocess = custom_preprocess
		self.custom_postprocess = custom_postprocess
	
	def on_train_begin(self, logs=None):
		self.deconv_model = DeconvolutionModel(self.model, self.img_array, self.custom_preprocess, self.custom_postprocess,
											   self.custom_keras_model_info)
	
	def on_batch_end(self, batch, logs=None):

		self.counter += 1

		# only update visualization at user specified intervals
		if self.counter == self.interval:

			# update weights
			self.deconv_model.update_deconv_model()

			# produce reconstructions
			reconstructions = self.deconv_model.produce_reconstructions_with_fixed_image(self.feat_map_layer_no,
																						 self.feat_map_amount,
																						 self.feat_map_nos)
			
			# save reconstructions as pickle
			with open(join(self.results_folder, 'deconvolution_network.pickle'), 'wb') as f:
				pickle.dump(reconstructions, f)
			
			self.counter = 0


class DeepVisualization(Callback):
	"""
	Based on http://yosinski.com/media/papers/Yosinski__2015__ICML_DL__Understanding_Neural_Networks_Through_Deep_Visualization__.pdf

	Learning rate and number of iterations are the bare minimum for creating a deep visualization image.
	With only these two values specified, a vanilla, or unregularized, visualization will be produced.
	l2_decay=[0.0, 1.0], blur_interval=[0, inf), blur_std=[0.0, inf), value_percentile=[0, 100], norm_percentile=[0, 100],
	contribution_percentile=[0, 100], and abs_contribution_percentile=[0, 100] are all regularization values.

	NOTE: both blur_interval and blur_std must be specified to enable Gaussian blurring

	Regularization technique explanations:

	- L2-decay is used to prevent a small number of extreme pixel values from dominating the output image

	- Gaussian blurring (with blur_interval = frequency of blurring, and blur_std = standard deviation for kernel)
	is used to penalize high frequency information in the output image.
	NOTE: standard deviation values between 0.0 and 0.3 work poorly, according to yosinski

	- Value percentile limit is used to induce sparsity by setting pixels with small absolute value to zero

	- Norm percentile limit is used to induce sparsity by setting pixels with small norm to zero

	- Contribution percentile limit is used to induce sparsity by setting pixels with small contribution to zero

	- Absolute contribution percentile limit is used to induce sparsity by setting pixels with small absolute contribution to zero
	"""

	# chosen units to visualize must be a list with elements on form tuple(layer number, unit index),
	# where unit index is tuple for layers with 3D structured output, like convolutional and pooling layers
	def __init__(self, file_folder, units_to_visualize, learning_rate, no_of_iterations, l2_decay=0, blur_interval=0,
				 blur_std=0, value_percentile=0, norm_percentile=0, contribution_percentile=0,
				 abs_contribution_percentile=0, custom_postprocess=None, interval=1000):
		
		super(DeepVisualization, self).__init__()
		
		# set channel dimension based on image data format from Keras backend
		if K.image_data_format() == 'channels_last':
			self.ch_dim = 3
		else:
			self.ch_dim = 1
		
		self.results_folder = join(file_folder, 'results')
		self.interval = interval
		self.counter = 0
		self.custom_postprocess = custom_postprocess
		
		# vanilla (required) values
		self.units_to_visualize = units_to_visualize
		self.learning_rate = learning_rate
		self.no_of_iterations = no_of_iterations
		
		# (ensemble) regularization values
		self.l2_decay = l2_decay
		self.blur_interval = blur_interval
		self.blur_std = blur_std
		self.value_percentile = value_percentile
		self.norm_percentile = norm_percentile
		self.contribution_percentile = contribution_percentile
		self.abs_contribution_percentile = abs_contribution_percentile
		
	def on_train_begin(self, logs=None):
		
		output_layer = self.model.layers[-1]
		output_layer_config = output_layer.get_config()
		
		# if top layer has softmax activation
		if output_layer_config['activation'] == 'softmax':
			# create a model similar to original, but with a linear activation in output layer, instead of softmax
			
			# alter activation in config to be linear
			output_layer_config['activation'] = 'linear'
			
			# create an alternative output layer using altered config and connect to same input layer as original output layer
			alt_output = type(output_layer).from_config(output_layer_config)(output_layer.input)

			# create the linear alternative model
			self.vis_model = Model(inputs=self.model.input, outputs=alt_output)
			
			# use weights from original output layer
			self.vis_model.layers[-1].set_weights(output_layer.get_weights())
		else:
			# if not, original model can be used
			self.vis_model = self.model
			
	def on_batch_end(self, batch, logs={}):

		self.counter += 1

		# only update visualization at user specified intervals
		if self.counter == self.interval:
			
			# list to hold visualization info for all chosen units
			vis_info = []
			
			# for the chosen layer number and unit index
			for layer_no, unit_index in self.units_to_visualize:
				
				# check if layer number is valid
				if layer_no < 0 or layer_no >= len(self.vis_model.layers):
					raise ValueError('Invalid layer number {}: Layer numbers should be between {} and {}'
									 .format(layer_no, 0, len(self.vis_model.layers) - 1))
				
				# create and save loss and gradient function for current unit
				compute_loss_and_gradients = self.get_loss_and_gradient_function(layer_no, unit_index)
				
				# create an initial visualization image
				visualization = self.create_initial_image()
				
				# perform gradient ascent update with regularization for n steps
				for i in range(1, self.no_of_iterations + 1):
					
					# compute loss and gradient values (input 0 as arg. #2 to deactivate training layers, like dropout)
					loss_value, pixel_gradients = compute_loss_and_gradients([visualization, 0])
					
					# update visualization image
					visualization += pixel_gradients * self.learning_rate
					
					# if regularization has been activated, regularize image
					visualization = self.apply_ensemble_regularization(visualization, pixel_gradients, i)
					
				# process visualization to match with standard image dimensions
				visualization = to_image_standard(visualization, self.custom_postprocess)
				
				# add to list of all visualization info
				# use self.model instead of self.vis_model to get original layer name if last layer
				vis_info.append((visualization, self.model.layers[layer_no].name, unit_index, loss_value))
				
			# save visualization images, complete with info about creation environment
			self.save_visualization_info(vis_info)
			
			self.counter = 0
	
	# regularizes input image with various techniques
	# each technique is activated by non-zero values for their respective variables
	def apply_ensemble_regularization(self, visualization, pixel_gradients, iteration_no):
		
		# regularizer #1
		# apply L2-decay
		if self.l2_decay > 0:
			visualization *= (1 - self.l2_decay)
		
		# regularizer #2
		# apply Gaussian blur
		if self.blur_interval > 0 and self.blur_std > 0:
			# only blur at certain iterations, as blurring is expensive
			if not iteration_no % self.blur_interval:
				# define standard deviations for blur kernel
				blur_kernel_std = [0, self.blur_std, self.blur_std, self.blur_std]
				
				# blur along height and width, but not along channel (color) dimension
				blur_kernel_std[self.ch_dim] = 0
				
				# perform blurring
				visualization = gaussian_filter(visualization, sigma=blur_kernel_std)
		
		# regularizer #3
		# apply value limit
		if self.value_percentile > 0:
			# find absolute values
			abs_visualization = abs(visualization)
			
			# find mask of high values (values above chosen value percentile)
			high_value_mask = abs_visualization >= np.percentile(abs_visualization, self.value_percentile)
			
			# apply to image to set pixels with small values to zero
			visualization *= high_value_mask
		
		# regularizer #4
		# apply norm limit
		if self.norm_percentile > 0:
			# compute pixel norms along channel (color) dimension
			pixel_norms = np.linalg.norm(visualization, axis=self.ch_dim)
			
			# find initial mask of high norms (norms above chosen norm percentile)
			high_norm_mask = pixel_norms >= np.percentile(pixel_norms, self.norm_percentile)
			
			# expand mask to account for color dimension
			high_norm_mask = self.expand_for_color(high_norm_mask)
			
			# apply to image to set pixels with small norms to zero
			visualization *= high_norm_mask
		
		# regularizer #5
		# apply contribution limit
		if self.contribution_percentile > 0:
			# predict the contribution of each pixel
			predicted_contribution = -visualization * pixel_gradients
			
			# sum over channel (color) dimension
			contribution = predicted_contribution.sum(self.ch_dim)
			
			# find initial mask of high contributions (contr. above chosen contr. percentile)
			high_contribution_mask = contribution >= np.percentile(contribution, self.contribution_percentile)
			
			# expand mask to account for color dimension
			high_contribution_mask = self.expand_for_color(high_contribution_mask)
			
			# apply to image to set pixels with small contributions to zero
			visualization *= high_contribution_mask
		
		# regularizer #6
		# apply absolute contribution limit
		if self.abs_contribution_percentile > 0:
			# predict the contribution of each pixel
			predicted_contribution = -visualization * pixel_gradients
			
			# sum over channel (color) dimension, and find absolute value
			abs_contribution = abs(predicted_contribution.sum(self.ch_dim))
			
			# find initial mask of high absolute contributions (abs. contr. above chosen abs. contr. percentile)
			high_abs_contribution_mask = abs_contribution >= np.percentile(abs_contribution,
																		   self.abs_contribution_percentile)
			
			# expand mask to account for color dimension
			high_abs_contribution_mask = self.expand_for_color(high_abs_contribution_mask)
			
			# apply to image to set pixels with small absolute contributions to zero
			visualization *= high_abs_contribution_mask
		
		return visualization
	
	# returns a function for computing loss and gradients of the loss for the chosen unit in the chosen layer w.r.t. the input image
	def get_loss_and_gradient_function(self, layer_no, unit_index):
	
		# if unit index is specified as integer, convert to tuple
		if isinstance(unit_index, int):
			unit_index = (unit_index,)
		
		# get output tensor based on chosen layer number
		output_tensor = self.vis_model.layers[layer_no].output
		
		# check that unit index is the correct length and that content is valid
		if len(output_tensor.shape[1:]) != len(unit_index):
			raise ValueError('Index mismatch: Unit indices should be of length {}, not {}'
							 .format(len(output_tensor.shape[1:]), len(unit_index)))
		else:
			tensor_min = np.array([0 for _ in output_tensor.shape[1:]])
			tensor_max = np.array([int(dim) - 1 for dim in output_tensor.shape[1:]])
			if np.any(np.array(unit_index) < tensor_min) or np.any(np.array(unit_index) > tensor_max):
				raise ValueError('Invalid unit index {}: Unit indices should have values between {} and {}'
								 .format(np.array(unit_index), tensor_min, tensor_max))
		
		# pad with batch index
		unit_index = (0,) + unit_index
			
		# loss is the activation of the unit in the chosen output tensor (chosen layer output)
		loss = output_tensor[unit_index]
		
		# compute gradients of the loss of the chosen unit w.r.t. the input image
		gradients = K.gradients(loss, self.vis_model.input)[0]
		
		# return function returning the loss and gradients given a visualization image
		# add a flag to disable the learning phase
		return K.function([self.vis_model.input, K.learning_phase()], [loss, gradients])
	
	# use to expand a (batch, height, width)-numpy array with a channel (color) dimension
	def expand_for_color(self, np_array):
		
		# expand at channel (color) dimension
		np_array = np.expand_dims(np_array, axis=self.ch_dim)
		
		# create tile repetition list, repeating in channel (color) dimension
		tile_reps = [1, 1, 1, 1]
		tile_reps[self.ch_dim] = self.vis_model.input_shape[self.ch_dim]
		
		# apply tile repetition
		np_array = np.tile(np_array, tile_reps)
		
		return np_array
	
	# creates an random, initial image to manipulate into a visualization
	def create_initial_image(self):
		
		# TODO: remove when done with testing
		# set random seed to be able to reproduce initial state of image
		# used in testing only, and should be remove upon implementation with tool
		np.random.seed(1337)
		
		# add (1,) for batch dimension
		return np.random.normal(0, 10, (1,) + self.vis_model.input_shape[1:])
	
	# saves the visualization and a txt-file describing its creation environment
	def save_visualization_info(self, vis_info):
		
		# create a txt-file containing information about creation environment
		env_info = 'The visualizations saved in deep_visualization.pickle were created using the following hyperparameters:\n\n' \
				   'Learning rate: {}\n' \
				   'Number of iterations: {}\n' \
				   '----------\n' \
				   'Regularization values\n\n' \
				   'L2-decay: {}\n' \
				   'Blur interval and std: {} & {}\n' \
				   'Value percentile: {}\n' \
				   'Norm percentile: {}\n' \
				   'Contribution percentile: {}\n' \
				   'Abs. contribution percentile: {}\n' \
				   ''.format(self.learning_rate, self.no_of_iterations, self.l2_decay, self.blur_interval,
							  self.blur_std, self.value_percentile, self.norm_percentile, self.contribution_percentile,
							  self.abs_contribution_percentile)

		# write creation environment info to text file
		with open(join(self.results_folder, 'deep_vis_env_info.txt'), 'w') as f:
			f.write(env_info)
	
		# write visualization info to pickle file
		with open(join(self.results_folder, 'deep_visualization.pickle'), 'wb') as f:
			pickle.dump(vis_info, f)


# utility function used to convert an array into a savable image array
def to_image_standard(img_array, custom_postprocess):

	# remove batch dimension, and alter color dimension accordingly
	img_array = img_array[0]

	if K.image_data_format() == 'channels_first':
		# alter dimensions from (color, height, width) to (height, width, color)
		img_array = img_array.transpose((1, 2, 0))

	if custom_postprocess is not None:
		img_array = custom_postprocess(img_array)

	# clip in [0, 255], and convert to uint8
	img_array = np.clip(img_array, 0, 255).astype('uint8')

	return img_array
