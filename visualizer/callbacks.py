import numpy as np
import pickle
import math

from os import mkdir, listdir
from os.path import join, basename

from scipy.misc import toimage
from scipy.ndimage.filters import gaussian_filter

from PIL import Image

import keras.backend as K
from keras.models import Model
from keras.callbacks import Callback
from keras.preprocessing import image

from visualizer.custom_keras_models import DeconvolutionModel
from keras.layers import Dropout
from keras.layers import Flatten


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

	def on_train_begin(self, logs={}):
		self.batches_in_epoch = math.ceil(self.params['samples'] / self.params['batch_size'])
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
			f.write("{0} {1} {2}\n".format(batch / self.batches_in_epoch, logs['acc'], logs['loss']))

	def on_epoch_end(self, epoch, logs={}):
		if self.params['do_validation']:
			with open(join(self.results_folder, 'training_progress_val.txt'), 'a') as f:
				# saves validation accuracy at each finished training epoch
				f.write("{0} {1} {2}\n".format(epoch, logs['val_acc'], logs['val_loss']))


# saves accuracy at each finished training batch
class Accuracy(Callback):

	def __init__(self, file_folder):
		super(Accuracy, self).__init__()
		self.results_folder = join(file_folder, 'results')
		self.batches_in_epoch = None

	def on_train_begin(self, logs={}):
		self.batches_in_epoch = math.ceil(self.params['samples'] / self.params['batch_size'])
		# ensure file creation
		with open(join(self.results_folder, 'accuracy_train.txt'), 'w') as f:
			f.write('')
		if self.params['do_validation']:
			with open(join(self.results_folder, 'accuracy_val.txt'), 'w') as f:
				f.write('')

	def on_batch_end(self, batch, logs={}):
		# write new accuracy line
		with open(join(self.results_folder, 'accuracy_train.txt'), 'a') as f:
			# saves accuracy at each finished training batch as a tuple of (x, y) values
			f.write(str(batch/self.batches_in_epoch) + ' ' + str(logs['acc']) + '\n')

	def on_epoch_end(self, epoch, logs={}):
		if self.params['do_validation']:
			with open(join(self.results_folder, 'accuracy_val.txt'), 'a') as f:
				# saves validation accuracy at each finished training epoch
				f.write(str(epoch + 1) + ' ' + str(logs['val_acc']) + '\n')


class Loss(Callback):

	def __init__(self, file_folder):
		super(Loss, self).__init__()
		self.results_folder = join(file_folder, 'results')
		self.batches_in_epoch = None

	def on_train_begin(self, logs={}):
		self.batches_in_epoch = math.ceil(self.params['samples'] / self.params['batch_size'])
		# ensure file creation
		with open(join(self.results_folder, 'loss_train.txt'), 'w') as f:
			f.write('')
		if self.params['do_validation']:
			with open(join(self.results_folder, 'loss_val.txt'), 'w') as f:
				f.write('')

	def on_batch_end(self, batch, logs={}):
		# write new loss line
		with open(join(self.results_folder, 'loss_train.txt'), 'a') as f:
			# saves accuracy at each finished training batch as a tuple of (x, y) values
			f.write(str(batch/self.batches_in_epoch) + ' ' + str(logs['loss']) + '\n')

	def on_epoch_end(self, epoch, logs={}):
		if self.params['do_validation']:
			with open(join(self.results_folder, 'loss_val.txt'), 'a') as f:
				# saves validation accuracy at each finished training epoch
				f.write(str(epoch + 1) + ' ' + str(logs['val_loss']) + '\n')


# saves activation arrays for each layer as tuples: (layer-name, array)
class LayerActivations(Callback):

	input_tensor = None

	def __init__(self, file_folder, interval=10, exclude_layers=None):

		super(LayerActivations, self).__init__()
		self.results_folder = join(file_folder, 'results')
		self.interval = interval
		self.counter = 0

		# needed to overcome mutable default arguments
		if exclude_layers is None:
			exclude_layers = [Dropout, Flatten]
		self.exclude_layers = exclude_layers

		# find image uploaded by user to use in visualization
		images_folder = join(file_folder, 'images')
		img_name = listdir(images_folder)[-1]
		img = Image.open(join(images_folder, img_name))

		# set input tensor and reshape to (1, width, height, 1)
		self.input_tensor = np.array(img)[np.newaxis, :, :, np.newaxis]

	def on_batch_end(self, batch, logs={}):

		# initialize layer tuple list with image
		layer_tuples = []

		# only update visualization at user specified intervals
		if self.counter == self.interval:

			for layer_no in range(len(self.model.layers)):

				layer = self.model.layers[layer_no]

				# check if layer should be included
				if not isinstance(layer, tuple(self.exclude_layers)):
					# create function using keras-backend for getting activation tensor
					get_activation_tensor = K.function([self.model.input, K.learning_phase()], [layer.output])

					# NOTE: learning phase 0 is testing and 1 is training (difference unknown as this point)

					tensor = get_activation_tensor([self.input_tensor, 0])[0][0]
					# scale to fit between [0.0, 255.0]
					if tensor.max() != 0.0:
						tensor *= (255.0 / tensor.max())

					if len(tensor.shape) == 3:
						# get on correct format (list of filters)
						tensor = np.rollaxis(tensor, 2)

					# save tuple (layer name, layer's activation tensor)
					layer_tuples.append(("Layer {0}: {1}".format(layer_no, layer.name), tensor))

			with open(join(self.results_folder, 'layer_activations.pickle'), 'wb') as f:
				pickle.dump(layer_tuples, f)

			self.counter = 0

		self.counter += 1


class SaliencyMaps(Callback):

	def __init__(self, file_folder, interval=10):
		super(SaliencyMaps, self).__init__()
		self.results_folder = join(file_folder, 'results')
		self.interval = interval
		self.counter = 0

		# find image uploaded by user to use in visualization
		images_folder = join(file_folder, 'images')
		img_name = listdir(images_folder)[-1]
		img = Image.open(join(images_folder, img_name))

		# convert to correct format TODO: check if this is needed, and generalize
		self.input_tensor = np.array(img)[np.newaxis, :, :, np.newaxis]

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

			with open(join(self.results_folder, 'saliency_maps.pickle'), 'wb') as f:
				pickle.dump(abs_saliency, f)

			self.counter = 0


class Deconvolution(Callback):
	def __init__(self, file_folder, feat_map_layer_no, feat_map_amount=None, feat_map_nos=None, interval=100):
		super(Deconvolution, self).__init__()
		
		self.results_folder = join(file_folder, 'results')
		self.interval = interval
		self.counter = 0
		
		# find image uploaded by user to use in visualization
		images_folder = join(file_folder, 'images')
		img_name = listdir(images_folder)[-1]
		pil_img = Image.open(join(images_folder, img_name))
		
		# convert to array and add batch dimension
		self.img = np.expand_dims(image.img_to_array(pil_img), axis=0)
		
		self.feat_map_layer_no = feat_map_layer_no
		self.feat_map_amount = feat_map_amount
		self.feat_map_nos = feat_map_nos
	
	def on_train_begin(self, logs=None):
		self.deconv_model = DeconvolutionModel(self.model, self.img)
	
	def on_batch_end(self, batch, logs=None):
		# only update visualization at user specified intervals
		if self.counter == self.interval:
			reconstructions = self.deconv_model.produce_reconstructions_with_fixed_image(self.feat_map_layer_no,
																						 self.feat_map_amount,
																						 self.feat_map_nos)
			
			# save reconstructions as pickle
			with open(join(self.results_folder, 'deconvolution.pickle'), 'wb') as f:
				pickle.dump(reconstructions, f)
			
			self.counter = 0
		
		self.counter += 1


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

	# chosen neurons to visualize must be a list with elements on form tuple(layer number, neuron number)
	def __init__(self, file_folder, neurons_to_visualize, learning_rate, no_of_iterations, l2_decay=0, blur_interval=0,
				 blur_std=0, value_percentile=0, norm_percentile=0, contribution_percentile=0,
				 abs_contribution_percentile=0, interval=1000):
		
		super(DeepVisualization, self).__init__()
		
		# set channel dimension based on image data format from Keras backend
		if K.image_data_format() == 'channels_last':
			self.ch_dim = 3
		else:
			self.ch_dim = 1
		
		self.results_folder = join(file_folder, 'results')
		self.interval = interval
		self.counter = 0
		
		# vanilla (required) values
		self.neurons_to_visualize = neurons_to_visualize
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

		# only update visualization at user specified intervals
		if self.counter == self.interval:
			
			# list to hold visualization info for all chosen neurons
			vis_info = []
			
			# for the chosen layer number and neuron number
			for layer_no, neuron_no in self.neurons_to_visualize:
				
				# create and save loss and gradient function for current neuron
				compute_loss_and_gradients = self.get_loss_and_gradient_function(layer_no, neuron_no)
				
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
				visualization = self.deprocess(visualization)
				
				# add to list of all visualization info
				vis_info.append((visualization, layer_no, neuron_no, loss_value))
				
			# save visualization images, complete with info about creation environment
			self.save_visualization_info(vis_info)
			
			self.counter = 0

		self.counter += 1
	
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
	
	# returns a function for computing loss and gradients w.r.t. the activations for the chosen neuron in the chosen layer
	def get_loss_and_gradient_function(self, layer_no, neuron_no):
		
		# loss is the activation of the neuron in the output of the chosen layer
		loss = self.vis_model.layers[layer_no].output[0, neuron_no]
		
		# gradients are computed from the visualization given as input w.r.t. this loss
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
	
	# utility function used to convert an array into a savable image array
	def deprocess(self, vis_array):
		
		# remove batch dimension, and alter color dimension accordingly
		img_array = vis_array[0]
		
		if K.image_data_format() == 'channels_first':
			# alter dimensions from (color, height, width) to (height, width, color)
			img_array = img_array.transpose((1, 2, 0))
		
		# clip in [0, 255], and convert to uint8
		img_array = np.clip(img_array, 0, 255).astype('uint8')
		
		return img_array
	
	# TODO: delete image saving part (modify, but don't delete info text) when done with testing
	# saves the visualization and a txt-file describing its creation environment
	def save_visualization_info(self, vis_info):
		
		# to hold easily readable information about visualizations' creation environments
		env_info = ''
		
		for vis_array, layer_no, neuron_no, loss_value in vis_info:
		
			# create appropriate name to identify image
			img_name = 'deep_vis_{}_{}'.format(layer_no, neuron_no)
			
			# TODO: delete when visualization on website is confirmed
			# process image to be saved
			img_to_save = vis_array.copy()
			# use self.ch_dim - 1 as we have removed batch dimension
			if img_to_save.shape[self.ch_dim - 1] == 1:
				# if greyscale image, remove inner dimension before save
				if K.image_data_format() == 'channels_last':
					img_to_save = img_to_save.reshape((img_to_save.shape[0], img_to_save.shape[1]))
				else:
					img_to_save = img_to_save.reshape((img_to_save.shape[1], img_to_save.shape[2]))
			
			# save the resulting image to disk
			# avoid scipy.misc.imsave because it will normalize the image pixel value between 0 and 255
			toimage(img_to_save).save(join(self.results_folder, img_name + '.png'))
			
			# also save a txt-file containing information about creation environment and obtained loss
			env_info += 'Image "{}.png" was created from neuron {} in layer {}, using the following hyperparameters:\n\n' \
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
						'----------\n' \
						'Obtained loss value: {}\n\n\n\n' \
						''.format(img_name, neuron_no, layer_no, self.learning_rate, self.no_of_iterations, self.l2_decay,
								  self.blur_interval, self.blur_std, self.value_percentile, self.norm_percentile,
								  self.contribution_percentile, self.abs_contribution_percentile, loss_value)

		# write creation environment info to text file
		with open(join(self.results_folder, 'deep_vis_env_info.txt'), 'w') as f:
			f.write(env_info)
	
		# write visualization info to pickle file
		with open(join(self.results_folder, 'deep_vis.pickle'), 'wb') as f:
			pickle.dump(vis_info, f)
