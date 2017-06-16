import numpy as np
import tensorflow as tf
import theano.tensor as tht
import scipy.misc

from time import time
from os.path import dirname, join

import keras.backend as K
from keras.models import Model
from keras.layers import Layer, Input, InputLayer, Conv2D, MaxPooling2D, Activation, Conv2DTranspose
from keras.preprocessing import image

# for images from URLs
import requests
from requests.exceptions import RequestException
from PIL import Image
from io import BytesIO

# define layers from which we can create a deconvolutional model
USABLE_LAYERS = (InputLayer, Conv2D, MaxPooling2D)

# define path to image URLS
urls_path = join(dirname(__file__), 'deconv_input', 'fall11_urls.txt')


class DeconvolutionalModel:
	def __init__(self, link_model, input_img, custom_preprocess=None, custom_postprocess=None, custom_keras_model_info=None):
		
		# set dimensions indices for rows, columns and channels
		if K.image_data_format() == 'channels_last':
			self.ch_dim = 3
		else:
			self.ch_dim = 1
		
		self.link_model = link_model
		self.custom_preprocess = custom_preprocess
		self.custom_postprocess = custom_postprocess
		self.input_img = self.preprocess_img(input_img)
		
		if custom_keras_model_info is None:
			# custom deconvolutional Keras model info is not specified, try to create automatically
			self.deconv_keras_model, self.layer_map = self.create_deconv_keras_model()
			
			# custom update is then not specified
			self.custom_update = None
		else:
			if None in custom_keras_model_info:
				raise ValueError("'None'-value found in 'custom_keras_model_info'-tuple. Tuple should contain (in respective "
								 "order): a deconvolutional Keras model based on your original model, a dictionary mapping "
								 "from original model layer numbers to the corresponding deconv. model layer numbers, "
								 "and an update method for the deconv. model which returns new deconv. model and layer map "
								 "(if no update needed, input a method with pass).")
			# unpack custom keras model info tuple
			self.deconv_keras_model, self.layer_map, self.custom_update = custom_keras_model_info
	
		# print deconv info
		# print('\n***DECONVOLUTIONAL MODEL INFO***')
		# print('Deconv. input shape:', self.deconv_keras_model.input_shape)
		# print('Deconv. output shape:', self.deconv_keras_model.output_shape)
		# print('\nLayers in deconv. model:')
		# for layer in self.deconv_keras_model.layers:
		# 	print(layer.name)
	
	def create_deconv_keras_model(self):
		
		start_time = time()
		
		# create layer map between conv. model layers and deconv. model layers
		layer_map = {}
		
		# get info used to create unpooling layers
		deconv_start_layer_no, unpool_info = self.compute_model_info()
		
		# add first layer with output shape of conv. model as input shape
		dc_input = Input(shape=self.link_model.layers[deconv_start_layer_no].output_shape[1:])
		
		# examine linked model from the top down and add appropriate layers
		dc_layer_count = 1
		x = dc_input
		for layer_no in range(deconv_start_layer_no, -1, -1):
			
			layer = self.link_model.layers[layer_no]
			
			# if convolution layer in linked model
			if isinstance(layer, Conv2D):
				# add activation before transposed convolution layer
				x = Activation(layer.activation)(x)
				
				# add transposed convolution layer
				x = Conv2DTranspose(filters=layer.input_shape[self.ch_dim],
									kernel_size=layer.kernel_size,
									strides=layer.strides,
									padding=layer.padding,
									data_format=layer.data_format,
									dilation_rate=layer.dilation_rate,
									weights=[layer.get_weights()[0]],
									use_bias=False)(x)
				
				# update layer map
				# TODO: may need to remove the +1 (currently it makes the deconvolution skip RELU-layer on intermediate feature map reconstructions)
				layer_map[layer_no] = dc_layer_count + 1
				dc_layer_count += 2
			
			# if pooling layer in linked model
			elif isinstance(layer, MaxPooling2D):
				# get previously computed input and output for pooling layer in linked model
				pool_input, pool_output = unpool_info[layer_no]
				
				# add unpooling layer (custom)
				x = MaxUnpooling2D(pool_input=pool_input,
								   pool_output=pool_output,
								   pool_size=layer.pool_size,
								   strides=layer.strides,
								   padding=layer.padding)(x)
				
				# update layer map
				layer_map[layer_no] = dc_layer_count
				dc_layer_count += 1
			else:
				# print('\nFound layer in original model which is neither convolutional or pooling, with layer name: ' + layer.name)
				pass
		
		# print('\nTime to create deconv. model was {:.4f} seconds'.format(time() - start_time))
		
		# return model and layer map
		return Model(inputs=dc_input, outputs=x), layer_map
	
	# TODO: find efficient way to compute pooling input and output
	def compute_model_info(self):
		
		# create new dict{layer number: tuple(pooling input, pooling output)}
		unpool_info = {}
		
		# set start values to first layer and initial image input
		start_layer_no = 0
		start_input = self.input_img
		
		# so long we have consecutive layers that can be used to create deconvolutional model
		layer_no = 0
		while layer_no < len(self.link_model.layers) and isinstance(self.link_model.layers[layer_no], USABLE_LAYERS):
			
			# if MaxPooling2D layer, collect information needed to create corresponding MaxUnpooling2D layer
			if isinstance(self.link_model.layers[layer_no], MaxPooling2D):
				# compute input and output for pooling layers
				pool_input = self.compute_layer_input(self.link_model, layer_no, start_layer_no, start_input)
				pool_output = self.compute_layer_output(self.link_model, layer_no, layer_no, pool_input)
				
				# add to info dict
				unpool_info[layer_no] = (pool_input, pool_output)
				
				# update values to start next computation in the layer after the current pooling layer
				start_layer_no = layer_no + 1
				start_input = pool_output
				
			layer_no += 1
		
		# return last layer examined as start layer of deconvolutional model, and info needed for unpooling
		return layer_no - 1, unpool_info
	
	# update model with by creating new model with updated layers
	def update_deconv_model(self, new_img=None):
		if new_img is not None:
			self.input_img = new_img
		
		if self.custom_update is None:
			self.deconv_keras_model, self.layer_map = self.create_deconv_keras_model()
		else:
			self.deconv_keras_model, self.layer_map = self.custom_update()
	
	# either uses maximally activated feature maps or specified ones
	def produce_reconstructions_with_fixed_image(self, feat_map_layer_no, feat_map_amount=None, feat_map_nos=None):
		
		if feat_map_layer_no > np.max(list(self.layer_map.keys())):
			raise ValueError("'feat_map_layer_no' value of {} is outside range of deconvolutional model. Max value is {}. (Layers numbers are zero-indexed.)".format(feat_map_layer_no, np.max(list(self.layer_map.keys()))))

		feat_map_no_max = self.link_model.layers[feat_map_layer_no].output_shape[self.ch_dim]
		
		if feat_map_nos is None:
			if feat_map_amount is None:
				raise ValueError("Neither 'feat_map_amount' or 'feat_maps_nos' are specified. Specify at least one: "
								 "'feat_map_amount' for maximally activated feature maps or 'feat_maps_nos' for user "
								 "selected feature maps. Set 'feat_map_amount' to -1 to select all feature maps.")
			
			# if feat_map_amount is set to -1, select all feature maps
			elif feat_map_amount == -1:
				feat_map_amount = feat_map_no_max
			
			# get maximally activated feature maps of the specified amount
			feat_maps_tuples = self.get_max_feature_maps(feat_map_layer_no, feat_map_amount)
		
		else:
			# if feature map numbers are specified, check if valid
			invalid_nos = np.array(feat_map_nos)[np.array(feat_map_nos) >= feat_map_no_max]
			if invalid_nos.size != 0:
				raise ValueError("'feat_maps_nos' contains numbers that are too large. Max is {}. "
								 "The invalid numbers were: {}".format(feat_map_no_max - 1, list(invalid_nos)))
			
			feat_maps_tuples = []
			for feat_map_no in feat_map_nos:
				# get conv. model output (feat maps) for desired feature map layer
				feat_maps = self.compute_layer_output(self.link_model, feat_map_layer_no, 0, self.input_img)
				
				max_act, max_act_pos = self.get_max_activation_and_pos(feat_maps, feat_map_no)
				
				# preprocess feature maps with regard to max activation in chosen feature map
				processed_feat_maps = self.preprocess_feat_maps(feat_maps.shape, max_act, max_act_pos)
				
				feat_maps_tuples.append((feat_map_no, processed_feat_maps))

		reconstructions = []
		for feat_map_no, processed_feat_maps in feat_maps_tuples:
			
			# TODO: currently results in a RecursionError for Theano
			# feed to deconv. model to produce reconstruction
			reconstruction = self.compute_layer_output(self.deconv_keras_model, -1, self.layer_map[feat_map_layer_no],
													   processed_feat_maps)
			
			# perform postprocessing of reconstruction to get usable image
			img_array = self.postprocess_reconstruction(reconstruction)
			
			reconstructions.append((img_array, self.link_model.layers[feat_map_layer_no].name, feat_map_no))
			
		return reconstructions
	
	# either uses randomly chosen feature maps or specified ones
	def produce_reconstructions_from_top_images(self, feat_map_layer_no, check_amount, choose_amount,
												feat_map_amount=None, feat_map_nos=None):
		
		if feat_map_layer_no > np.max(list(self.layer_map.keys())):
			raise ValueError("'feat_map_layer_no' value of {} is outside range of deconvolutional model. Max value is {}. "
							 "(Layers numbers are zero-indexed.)".format(feat_map_layer_no, np.max(list(self.layer_map.keys()))))

		feat_map_no_max = self.link_model.layers[feat_map_layer_no].output_shape[self.ch_dim]
		
		if feat_map_nos is None:
			if feat_map_amount is None:
				raise ValueError("Neither 'feat_map_amount' or 'feat_maps_nos' are specified. Specify at least one: "
								 "'feat_map_amount' for a random subset of feature maps or 'feat_maps_nos' for user "
								 "selected feature maps.")
			
			# TODO: delete SEED when done with testing
			np.random.seed(1337)
			
			# select random subset of feature map numbers of specified size
			feat_map_nos = np.random.choice(feat_map_no_max, feat_map_amount, replace=False)
		
		else:
			# if feature map numbers are specified, check if valid
			invalid_nos = np.array(feat_map_nos)[np.array(feat_map_nos) >= feat_map_no_max]
			if invalid_nos.size != 0:
				raise ValueError("'feat_maps_nos' contains numbers that are too large. Max is {}. "
								 "The invalid numbers were: {}".format(feat_map_no_max - 1, list(invalid_nos)))
		
		# print('\nReconstruct for feature maps in layer {}: {}'.format(feat_map_layer_no, feat_map_nos))
		max_images_dict, max_imgs_info_by_feat_map_no = self.get_max_images(check_amount, choose_amount, feat_map_layer_no, feat_map_nos)
		
		reconstructions_by_feat_map_no = {}
		for feat_map_no in feat_map_nos:
			reconstructions_by_feat_map_no[feat_map_no] = []
			counter = 0
			for max_img in max_images_dict[feat_map_no]:
				
				# update all unpooling layers for the new image
				self.update_deconv_model(max_img)
				
				# get conv. model output (feat maps) for desired feature map layer
				feat_maps = self.compute_layer_output(self.link_model, feat_map_layer_no, 0, self.input_img)
				
				max_act, max_act_pos = self.get_max_activation_and_pos(feat_maps, feat_map_no)
				
				# preprocess feature maps with regard to max activation in chosen feature map
				processed_feat_maps = self.preprocess_feat_maps(feat_maps.shape, max_act, max_act_pos)
				
				# feed to deconv. model to produce reconstruction
				reconstruction = self.compute_layer_output(self.deconv_keras_model, -1, self.layer_map[feat_map_layer_no],
														   processed_feat_maps)
				
				# perform postprocessing of reconstruction to get usable image
				img_array = self.postprocess_reconstruction(reconstruction)
				
				reconstructions_by_feat_map_no[feat_map_no].append((img_array, self.link_model.layers[feat_map_layer_no].name, counter))
				
				counter += 1
		
		return reconstructions_by_feat_map_no, max_imgs_info_by_feat_map_no
	
	def compute_layer_input(self, model, end_layer_no, start_layer_no, input_array):
		
		input_func = K.function([model.layers[start_layer_no].input, K.learning_phase()],
								[model.layers[end_layer_no].input])
		return input_func([input_array, 0])[0]
	
	def compute_layer_output(self, model, end_layer_no, start_layer_no, input_array):
		
		output_func = K.function([model.layers[start_layer_no].input, K.learning_phase()],
								 [model.layers[end_layer_no].output])
		
		return output_func([input_array, 0])[0]
	
	def get_max_activation_and_pos(self, feat_maps, feat_map_no):
		
		if K.image_data_format() == 'channels_last':
			# get selected feature map based on input
			selected_feat_map = feat_maps[:, :, :, feat_map_no]
			
			# get index for max element in given feature map
			max_activation_pos = np.unravel_index(np.argmax(selected_feat_map), selected_feat_map.shape)
			
			# expand with feature map dimension
			max_activation_pos += (feat_map_no,)
		else:
			# TODO: test support for theano
			# get selected feature map based on input
			selected_feat_map = feat_maps[:, feat_map_no, :, :]
			
			# get index for max element in given feature map
			max_activation_pos = np.unravel_index(np.argmax(selected_feat_map), selected_feat_map.shape)
			
			# expand with feature map dimension
			max_activation_pos = (max_activation_pos[0], feat_map_no, max_activation_pos[1], max_activation_pos[2])
		
		# return max activation and its position
		return feat_maps[max_activation_pos], max_activation_pos
	
	def preprocess_feat_maps(self, feat_maps_shape, max_activation, max_activation_pos):
		
		# set all entries except max activation of chosen feature map to zero
		processed_feat_maps = np.zeros(feat_maps_shape)
		processed_feat_maps[max_activation_pos] = max_activation
		
		return processed_feat_maps
	
	# find feature maps with largest single element values
	def get_max_feature_maps(self, feat_map_layer_no, amount):
		
		feat_maps = self.compute_layer_output(self.link_model, feat_map_layer_no, 0, self.input_img)
		
		max_activations = []
		max_positions = []
		
		# collect all feature map maximal activation
		for feat_map_no in range(feat_maps.shape[self.ch_dim]):
			max_act, max_act_pos = self.get_max_activation_and_pos(feat_maps, feat_map_no)
			
			max_activations.append(max_act)
			max_positions.append(max_act_pos)
		
		max_feat_maps_tuples = []
		for feat_map_no in np.array(max_activations).argsort()[-amount:][::-1]:
			# set all entries except max activation of chosen feature map to zero
			processed_feat_maps = self.preprocess_feat_maps(feat_maps.shape, max_activations[feat_map_no],
															max_positions[feat_map_no])
			
			max_feat_maps_tuples.append((feat_map_no, processed_feat_maps))
		
		return max_feat_maps_tuples
	
	# TODO: currently based on ImageNet '11 text file with image URLs
	def get_max_images(self, check_amount, choose_amount, feat_map_layer_no, feat_map_nos):
		
		urls = []
		image_scores = {}
		
		# initialize image_scores with empty list
		for feat_map_no in feat_map_nos:
			image_scores[feat_map_no] = []
		
		with open(urls_path, 'r') as f:
			for i in range(check_amount):
				url = f.readline().split('\t')[1].rstrip()
				
				# print('Check image ' + str(i) + ' at URL: ' + url)
				
				try:
					img_array = self.load_image_from_url(url)
				except RequestException:
					# print('Error with request')
					continue
				except ValueError:
					# print('Image no longer exists')
					continue
				except OSError:
					# print('Error in opening image')
					continue
				
				urls.append(url)
				
				feat_maps = self.compute_layer_output(self.link_model, feat_map_layer_no, 0, img_array)
				
				if K.image_data_format() == 'channels_last':
					for feat_map_no in feat_map_nos:
						image_scores[feat_map_no].append(np.amax(feat_maps[:, :, :, feat_map_no]))
				else:
					for feat_map_no in feat_map_nos:
						image_scores[feat_map_no].append(np.amax(feat_maps[:, feat_map_no, :, :]))
		
		chosen_images_dict = {}
		max_imgs_info_by_feat_map_no = {}
		for feat_map_no in feat_map_nos:
			chosen_images_dict[feat_map_no] = []
			max_imgs_info_by_feat_map_no[feat_map_no] = []
			# print('\nChosen image URLs for feat. map no. {}:'.format(feat_map_no))
			for index in np.array(image_scores[feat_map_no]).argsort()[-choose_amount:][::-1]:
				# print(urls[index])
				
				# if image was safely loaded the first time, assume safe to load now
				img_array = self.load_image_from_url(urls[index])
				chosen_images_dict[feat_map_no].append(img_array)
				
				# remove batch dimension, clip in [0, 255], and convert to uint8
				img_array = img_array[0]
				img_array = np.clip(img_array, 0, 255).astype('uint8')
				
				max_imgs_info_by_feat_map_no[feat_map_no].append((img_array, urls[index]))
			
		return chosen_images_dict, max_imgs_info_by_feat_map_no
	
	def preprocess_img(self, img_array):
		
		# apply custom preprocessing if supplied
		if self.custom_preprocess is not None:
			img_array = self.custom_preprocess(img_array)
			
		# expand with batch dimension
		img_array = np.expand_dims(img_array, 0)
		
		return img_array
		
	# NOTE: throws requests.exceptions.RequestException, ValueError, OSError
	def load_image_from_url(self, url):
		response = requests.get(url, timeout=5)
		
		img_array = image.img_to_array(Image.open(BytesIO(response.content)))
		img_array = self.preprocess_img(img_array)
		
		return img_array
	
	# TODO: modify when done with testing
	# processes and saves the reconstruction and returns processed array and name
	def postprocess_reconstruction(self, rec_array):
		
		# process reconstruction array
		# remove batch dimension
		rec_array = rec_array[0]
		
		if K.image_data_format() == 'channels_first':
			# alter dimensions from (color, height, width) to (height, width, color)
			rec_array = rec_array.transpose((1, 2, 0))
			
		if self.custom_postprocess is not None:
			rec_array = self.custom_postprocess(rec_array)
		
		# clip in [0, 255] and convert to uint8
		rec_array = rec_array.clip(0, 255).astype('uint8')
		
		return rec_array


# generates a recreated pooling input from pooling output and pooling configuration
# the recreated input is zero, except from entries where the pooling output entries where originally chosen from,
# where the value is the same as the corresponding pooling output entry
class MaxUnpooling2D(Layer):
	#########################################################
	### these three initial methods are required by Layer ###
	#########################################################
	
	def __init__(self, pool_input, pool_output, pool_size, strides, padding, **kwargs):
		
		# check backend to detect dimensions used
		if K.image_data_format() == 'channels_last':
			# tensorflow is used, dimension are (samples, rows, columns, channels)
			row_dim = 1
			col_dim = 2
			self.ch_dim = 3
		else:
			# theano is used, dimension are (samples, channels, rows, columns)
			row_dim = 2
			col_dim = 3
			self.ch_dim = 1
			
		self.pool_input = pool_input
		self.pool_output = pool_output
		
		# information about original pooling behaviour
		# pool size and strides are both (rows, columns)-tuples
		self.pool_size = pool_size
		self.strides = strides
		
		# if padding is same, use offsets to correct computed pooling regions (simulates padding)
		# initialize offset to 0, as it is not needed when padding is valid
		row_offset = 0
		col_offset = 0
		# padding is either 'valid' or 'same'
		if padding == 'same':
			
			if K.backend() == 'tensorflow':
				# when using tensorflow, padding is not always added, and a pooling region center is never in the padding.
				# if there is no obvious center in the pooling region, unlike in 3x3, the max pooling will use upper- and
				# leftmost entry of the possible center entries as the actual center, e.g. entry [1, 1] in a 4x4 region.
				# padding will be added if it is possible to fit another center within the original tensor. padding will
				# also be added to balance the how much of each region is inside the original tensor, e.g. padding to get
				# two regions with 75% of entries inside each instead of one region with 100% inside and the other with 25%
				# inside
				
				# find offset (to simulate padding) by computing total region space that falls outside of original tensor
				# and divide by two to distribute to top-bottom/left-right of original tensor
				row_offset = ((pool_output.shape[row_dim] - 1) * strides[0] + pool_size[0] - pool_input.shape[row_dim]) // 2
				col_offset = ((pool_output.shape[col_dim] - 1) * strides[1] + pool_size[1] - pool_input.shape[col_dim]) // 2
				
				# TODO: find alternative to these negative checks, seems to be produced when total stride length == length, and strides > pool size
				# computed offset can be negative, but this equals no offset
				if row_offset < 0:
					row_offset = 0
				if col_offset < 0:
					col_offset = 0
			else:
				# when using theano, padding is always added, and a pooling region center is never in the padding.
				# if there is no obvious center in the pooling region, unlike in 3x3, the max pooling
				# will use lower- and rightmost entry in the pooling region as the actual center, e.g. entry [3, 3] in a 4x4
				# region. if only the rows have an unclear center, the center is chosen to be the at the lowermost row and
				# the natural column center, e.g. entry [3, 1] in a 4x3 region. similarly, if only the columns have an
				# unclear center, the center is chosen to be at the natural row center and the rightmost column, e.g. [3, 1]
				# in a 3X4 region.
				
				# set offset (to simulate padding) to lowermost and rightmost entries by default
				row_offset = pool_size[0] - 1
				col_offset = pool_size[1] - 1
				
				# if rows have a clear center, update offset
				if pool_size[0] % 2 == 1:
					row_offset //= 2
				
				# if columns have a clear center, update offset
				if pool_size[1] % 2 == 1:
					col_offset //= 2
					
		# compute region indices for every element in pooling output
		self.region_indices = []
		for i in range(pool_output.shape[row_dim]):
			# compute pooling region row indices
			start_row, end_row = self.compute_index_interval(i, 0, row_offset, pool_input.shape[row_dim])
			
			for j in range(pool_output.shape[col_dim]):
				# compute pooling region column indices
				start_col, end_col = self.compute_index_interval(j, 1, col_offset, pool_input.shape[col_dim])
				
				self.region_indices.append((start_row, end_row, start_col, end_col, i, j))
		
		super(MaxUnpooling2D, self).__init__(**kwargs)
	
	def call(self, inputs):
		
		# for every region, find max indices
		max_indices = []
		# sample size is always 1, as unpooling has image specific switches
		sample_no = 0
		for start_row, end_row, start_col, end_col, i, j in self.region_indices:
			max_indices.extend([self.get_region_max_index(sample_no, start_row, end_row, start_col, end_col, i, j, channel)
								for channel in range(self.pool_input.shape[self.ch_dim])])
		
		if K.backend() == 'tensorflow':
			# use tensorflow
			recreated_input = tf.scatter_nd(indices=max_indices,
											updates=tf.reshape(inputs, [-1]),
											shape=self.pool_input.shape,
											name=self.name + '_output')
		else:
			# use theano
			# very inefficient
			recreated_input = tht.zeros(self.pool_input.shape)
			for input_index, output_index in max_indices:
				recreated_input = tht.set_subtensor(recreated_input[input_index], inputs[output_index])
		
		return recreated_input
	
	def compute_output_shape(self, input_shape):
		return (input_shape[0],) + self.pool_input.shape[1:]
	
	##############################################
	### what follows are custom helper methods ###
	##############################################

	def get_region_max_index(self, sample_no, start_row, end_row, start_col, end_col, i, j, channel):

		if K.backend() == 'tensorflow':
			# use tensorflow dimensions
			for row in range(start_row, end_row):
				for col in range(start_col, end_col):
					if self.pool_input[sample_no, row, col, channel] == self.pool_output[sample_no, i, j, channel]:
						return (sample_no, row, col, channel)
		else:
			# use theano dimensions
			for row in range(start_row, end_row):
				for col in range(start_col, end_col):
					if self.pool_input[sample_no, channel, row, col] == self.pool_output[sample_no, channel, i, j]:
						return ((sample_no, channel, row, col), (sample_no, channel, i, j))
	
	# computes index intervals for pooling regions, and is applicable for both row and column indices
	# considering pooling regions as entries in a traditional matrix, the region_index describes either a row or column
	# index for the region entry in such a matrix
	# tuple_no describes where in the strides and pool size tuples one should get values from, with row values at tuple
	# index 0 and column values are at tuple index 1
	def compute_index_interval(self, region_index, tuple_no, offset, max_index):
		
		start_index = region_index * self.strides[tuple_no] - offset
		end_index = start_index + self.pool_size[tuple_no]
		
		# we ignore padded parts, so if offset makes start index smaller than smallest index in original pooling input
		# or end index larger than largest index in original pooling input, correct
		if start_index < 0:
			start_index = 0
		if end_index > max_index:
			end_index = max_index
		
		return start_index, end_index
