import numpy as np
import tensorflow as tf
import theano.tensor as tht
import scipy.misc

from time import time
from os import mkdir, listdir
from os.path import dirname, join
from requests.exceptions import RequestException

import keras.backend as K
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Activation, Conv2DTranspose
from keras.layers.pooling import _Pooling2D
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image

# TODO: delete when done with testing
# for images from URLs
from PIL import Image
import requests
from io import BytesIO

# VGG16 mean values
MEAN_VALUES = np.array([103.939, 116.779, 123.68])

# define output path and make folder
output_path = join(dirname(__file__), 'deconv_output')

# define path to image URLS
urls_path = join(dirname(__file__), 'deconv_input', 'fall11_urls.txt')


# TODO: delete when done with testing
# NOTE: throws requests.exceptions.RequestException, ValueError, OSError
def load_image_from_url(url):
	response = requests.get(url, timeout=5)

	img = Image.open(BytesIO(response.content))
	img = image.img_to_array(img)
	img = scipy.misc.imresize(img, (224, 224))
	img = img.astype('float64')

	return preprocess_image(img)


def load_image_from_file(img_name):
	img_path = join(dirname(__file__), 'input', img_name)
	img = image.load_img(img_path, target_size=(224, 224))
	img = image.img_to_array(img)

	return preprocess_image(img)


def preprocess_image(img):

	if K.image_data_format() == 'channels_last':
		img -= MEAN_VALUES.reshape((1, 1, 3))
	else:
		img -= MEAN_VALUES.reshape((3, 1, 1))

	return np.expand_dims(img, axis=0)


# utility function used to convert a tensor into a savable image
def tensor_to_img(tensor):
	# remove batch dimension
	img = tensor[0]

	if K.image_data_format() == 'channels_first':
		# alter dimensions from (color, height, width) to (height, width, color)
		img = img.transpose((1, 2, 0))

	img += MEAN_VALUES.reshape((1, 1, 3))

	img = np.clip(img, 0, 255).astype('uint8')  # clip in [0;255] and convert to int

	return img


# saves the visualization and a txt-file describing its creation environment
def save_reconstruction(recon, feat_map_no, fixed_image_used=True):
	# process before save
	img = tensor_to_img(recon)

	if fixed_image_used:
		image_name = 'max_no_{}_feat_map_{}'.format(len(listdir(output_path)) + 1, feat_map_no)
	else:
		image_name = 'feat_map_{}_recon_{}'.format(feat_map_no, len(listdir(output_path)) + 1)

	# save the resulting image to disk
	scipy.misc.toimage(img, cmin=0, cmax=255).save(join(output_path, image_name + '.png'))
	# avoid scipy.misc.imsave because it will normalize the image pixel value between 0 and 255

	# print('\nImage has been saved as {!s}.png\n'.format(image_name))


def deconv_example():
	img_name = 'dog.jpg'
	img = load_image_from_file(img_name)

	conv_model = VGG16(include_top=False, weights='imagenet', input_shape=img.shape[1:])

	# print conv info
	# print('\n***CONVOLUTIONAL MODEL INFO***')
	# print('Conv. input shape:', conv_model.input_shape)
	# print('Conv. output shape:', conv_model.output_shape)
	# print('\nLayers in conv. model:')
	# for layer in conv_model.layers:
	# 	print(layer.name)

	print('\nCreating deconvolution model')
	start_time = time()
	deconv_model = DeconvolutionModel(conv_model, img, img_name)
	print('\nTime to create was {:.4f} seconds'.format(time() - start_time))

	# note that layers are zero indexed
	feat_map_layer_no = 18

	choose_max_images = False

	print('\nReady for deconv. pred.')
	start_time = time()
	
	if choose_max_images:
		deconv_model.produce_reconstruction_from_top_images(feat_map_layer_no, 100, 5, 3)
		# deconv_model.produce_reconstruction_from_top_images(feat_map_layer_no, 100, 5, feat_map_nos=[88, 351, 178])
	else:
		deconv_model.produce_reconstruction_with_fixed_image(feat_map_layer_no, 10)
		# deconv_model.produce_reconstruction_with_fixed_image(feat_map_layer_no, feat_map_nos=[88, 351, 178, 0, 5])
		
	print('\nTime to perform reconstructions for feat maps was {:.4f} seconds'.format(time() - start_time))


class DeconvolutionModel:
	def __init__(self, link_model, input_img, input_img_name):
		
		# TODO: test with img-model map
		self.model_map = {}
		
		# set dimensions indices for rows, columns and channels
		if K.image_data_format() == 'channels_last':
			self.ch_dim = 3
		else:
			self.ch_dim = 1
		
		self.link_model = link_model
		self.input_img = input_img
		self.input_img_name = input_img_name
		
		self.deconv_model, self.layer_map = self.create_deconv_model()
	
		# print deconv info
		# print('\n***DECONVOLUTIONAL MODEL INFO***')
		# print('Deconv. input shape:', self.deconv_model.input_shape)
		# print('Deconv. output shape:', self.deconv_model.output_shape)
		# print('\nLayers in deconv. model:')
		# for layer in self.deconv_model.layers:
		# 	print(layer.name)
	
	def create_deconv_model(self):
		
		start_time = time()
		
		try:
			deconv_model, layer_map = self.model_map[self.input_img_name]
		except KeyError:
			
			# create layer map between conv. layers and deconv. layers
			layer_map = {}

			# get info used to create unpooling layers
			unpool_info = self.get_unpool_info()

			# add first layer with output shape of conv. model as input shape
			dc_input = Input(shape=self.link_model.output_shape[1:])
			
			# examine linked model from the top down and add appropriate layers
			dc_layer_count = 1
			x = dc_input
			for layer_no in range(len(self.link_model.layers) - 1, -1, -1):
				
				layer = self.link_model.layers[layer_no]
				
				# if convolution layer in linked model
				if isinstance(layer, Conv2D):
					# add activation before deconvolution layer
					x = Activation(layer.activation)(x)
					
					# add deconvolution layer (called Conv2DTranspose in Keras)
					x = Conv2DTranspose(filters=layer.input_shape[self.ch_dim],
										kernel_size=layer.kernel_size,
										strides=layer.strides,
										padding=layer.padding,
										data_format=layer.data_format,
										dilation_rate=layer.dilation_rate,
										# weights=flip_weights(layer.get_weights()),
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
			
			# create model
			deconv_model = Model(inputs=dc_input, outputs=x)
			
			# add to model map
			self.model_map[self.input_img_name] = (deconv_model, layer_map)
		
		# print('\nTime to create deconv. model was {:.4f} seconds'.format(time() - start_time))
		
		return deconv_model, layer_map
	
	# TODO: find efficient way to compute pooling input and output
	def get_unpool_info(self):
		
		# create new dict{layer number: tuple(pooling input, pooling output)}
		unpool_info = {}
		
		# set start values to first layer and initial image input
		start_layer_no = 0
		start_input = self.input_img
		
		# traverse layers and compute input and output for pooling layers
		for layer_no in range(len(self.link_model.layers)):
			if isinstance(self.link_model.layers[layer_no], MaxPooling2D):
				pool_input, pool_output = self.compute_layer_input_and_output(self.link_model, layer_no, start_layer_no,
																			  start_input)
				
				# add to info dict
				unpool_info[layer_no] = (pool_input, pool_output)
				
				# update values to start next computation in the layer after the current pooling layer
				start_layer_no = layer_no + 1
				start_input = pool_output
		
		return unpool_info
	
	# update model with by creating new model with updated unpooling layers (unpooling is image specific)
	def update_deconv_model(self, new_img, new_img_name):
		self.input_img = new_img
		self.input_img_name = new_img_name
		
		self.deconv_model, self.layer_map = self.create_deconv_model()
	
	# either uses maximally activated feature maps or specified ones
	def produce_reconstruction_with_fixed_image(self, feat_map_layer_no, feat_map_amount=None, feat_map_nos=None):
		
		if feat_map_nos is None:
			if feat_map_amount is None:
				raise ValueError("Neither 'feat_map_amount' or 'feat_maps_nos' are specified. Specify at least one: "
								 "'feat_map_amount' for maximally activated feature maps or 'feat_maps_nos' for user "
								 "selected feature maps.")
			
			# get maximally activated feature maps of the specified amount
			feat_maps_tuples = self.get_max_feature_maps(feat_map_layer_no, feat_map_amount)
				
		else:
			# if feature map numbers are specified, check if valid
			feat_map_no_max = self.link_model.layers[feat_map_layer_no].output_shape[self.ch_dim]
			invalid_nos = np.array(feat_map_nos)[np.array(feat_map_nos) >= feat_map_no_max]
			if invalid_nos.size != 0:
				raise ValueError("'feat_maps_nos' contains numbers that are too large. Max is {}. "
								 "The invalid numbers were: {}".format(feat_map_no_max - 1, list(invalid_nos)))
			
			feat_maps_tuples = []
			for feat_map_no in feat_map_nos:
				# get conv. model output (feat maps) for desired feature map layer
				_, feat_maps = self.compute_layer_input_and_output(self.link_model, feat_map_layer_no, 0,
																   self.input_img)
				
				max_act, max_act_pos = self.get_max_activation_and_pos(feat_maps, feat_map_no)
				
				# preprocess feature maps with regard to max activation in chosen feature map
				processed_feat_maps = self.preprocess_feat_maps(feat_maps.shape, max_act, max_act_pos)
				
				feat_maps_tuples.append((feat_map_no, processed_feat_maps))

		counter = 0
		for feat_map_no, processed_feat_maps in feat_maps_tuples:
			print(counter, feat_map_no)
			
			# feed to deconv. model to produce reconstruction
			_, reconstruction = self.compute_layer_input_and_output(self.deconv_model, -1,
																	self.layer_map[feat_map_layer_no],
																	processed_feat_maps)
			# save reconstruction to designated folder
			save_reconstruction(reconstruction, feat_map_no)

			counter += 1

	# either uses randomly chosen feature maps or specified ones
	def produce_reconstruction_from_top_images(self, feat_map_layer_no, check_amount, choose_amount, feat_map_amount=None, feat_map_nos=None):
		
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
			
		print('\nReconstruct for feature maps in layer {}: {}'.format(feat_map_layer_no, feat_map_nos))
		max_images_dict, urls_dict = self.get_max_images(check_amount, choose_amount, feat_map_layer_no, feat_map_nos)
		
		for feat_map_no in feat_map_nos:
			for i in range(len(max_images_dict[feat_map_no])):
				max_img = max_images_dict[feat_map_no][i]
				max_img_name = urls_dict[feat_map_no][i]
				
				# update all unpooling layers for the new image
				self.update_deconv_model(max_img, max_img_name)
				
				# get conv. model output (feat maps) for desired feature map layer
				_, feat_maps = self.compute_layer_input_and_output(self.link_model, feat_map_layer_no, 0,
																   self.input_img)
				
				max_act, max_act_pos = self.get_max_activation_and_pos(feat_maps, feat_map_no)

				# preprocess feature maps with regard to max activation in chosen feature map
				processed_feat_maps = self.preprocess_feat_maps(feat_maps.shape, max_act, max_act_pos)
				
				# feed to deconv. model to produce reconstruction
				_, reconstruction = self.compute_layer_input_and_output(self.deconv_model, -1,
																		self.layer_map[feat_map_layer_no],
																		processed_feat_maps)
				
				# save reconstruction to designated folder
				save_reconstruction(reconstruction, feat_map_no, False)
	
	def compute_layer_input_and_output(self, model, end_layer_no, start_layer_no, start_input):
		input_func = K.function([model.layers[start_layer_no].input, K.learning_phase()],
								[model.layers[end_layer_no].input])
		layer_input = input_func([start_input, 0])[0]
		
		output_func = K.function([model.layers[end_layer_no].input, K.learning_phase()],
								 [model.layers[end_layer_no].output])
		layer_output = output_func([layer_input, 0])[0]
		
		return layer_input, layer_output
	
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
	def get_max_feature_maps(self, feat_map_layer_no, amount=-1):
		
		_, feat_maps = self.compute_layer_input_and_output(self.link_model, feat_map_layer_no, 0, self.input_img)
		
		# if negative number, select all
		if amount < 0:
			amount = feat_maps.shape[self.ch_dim]
			print('Amount updated to:', amount)
		
		max_activations = []
		max_positions = []
		
		# collect all feature map maximal activation
		for feat_map_no in range(feat_maps.shape[self.ch_dim]):
			
			max_act, max_act_pos = self.get_max_activation_and_pos(feat_maps, feat_map_no)

			max_activations.append(max_act)
			max_positions.append(max_act_pos)
		
		max_feat_maps_tuples = []
		counter = 0
		for feat_map_no in np.array(max_activations).argsort()[-amount:][::-1]:
			# set all entries except max activation of chosen feature map to zero
			processed_feat_maps = self.preprocess_feat_maps(feat_maps.shape, max_activations[feat_map_no], max_positions[feat_map_no])
		
			max_feat_maps_tuples.append((feat_map_no, processed_feat_maps))
		
			if max_activations[feat_map_no] < 0.01:
				counter += 1
		
		print('There were {} minor activations among those chosen'.format(counter))
		
		return max_feat_maps_tuples
	
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
					img = load_image_from_url(url)
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
				
				_, feat_maps = self.compute_layer_input_and_output(self.link_model, feat_map_layer_no, 0, img)
				
				if K.image_data_format() == 'channels_last':
					for feat_map_no in feat_map_nos:
						image_scores[feat_map_no].append(np.amax(feat_maps[:, :, :, feat_map_no]))
				else:
					for feat_map_no in feat_map_nos:
						image_scores[feat_map_no].append(np.amax(feat_maps[:, feat_map_no, :, :]))
		
		chosen_urls_dict = {}
		chosen_images_dict = {}
		for feat_map_no in feat_map_nos:
			chosen_urls_dict[feat_map_no] = []
			chosen_images_dict[feat_map_no] = []
			count = 1
			print('\nChosen image URLs for feat. map no. {}:'.format(feat_map_no))
			for index in np.array(image_scores[feat_map_no]).argsort()[-choose_amount:][::-1]:
				print(urls[index])
				chosen_urls_dict[feat_map_no].append(urls[index])
				
				# if image was safely loaded the first time, assume safe to load now
				img = load_image_from_url(urls[index])
				chosen_images_dict[feat_map_no].append(img)
				
				# save max images for comparison
				img = img[0]
				img = np.clip(img, 0, 255).astype('uint8')  # clip in [0;255] and convert to int
				scipy.misc.toimage(img, cmin=0, cmax=255).save(
					join(dirname(__file__), 'deconv_max_images', 'feat_map_{}_max_image_{}_index_{}.png'.format(feat_map_no, count, index)))
				
				count += 1
		
		return chosen_images_dict, chosen_urls_dict


# generates a recreated pooling input from pooling output and pooling configuration
# the recreated input is zero, except from entries where the pooling output entries where originally chosen from,
# where the value is the same as the corresponding pooling output entry
class MaxUnpooling2D(_Pooling2D):
	#########################################################
	### these three initial methods are required by Layer ###
	#########################################################

	def __init__(self, pool_input, pool_output, pool_size, strides, padding, **kwargs):

		# check backend to detect dimensions used
		if K.image_data_format() == 'channels_last':
			# tensorflow is used, dimension are (samples, rows, columns, channels)
			self.row_dim = 1
			self.col_dim = 2
			self.ch_dim = 3
		else:
			# theano is used, dimension are (samples, channels, rows, columns)
			self.row_dim = 2
			self.col_dim = 3
			self.ch_dim = 1

		# create placeholder values for pooling input and output
		self.pool_input = pool_input
		self.pool_output = pool_output

		# information about original pooling behaviour
		# pool size and strides are both (rows, columns)-tuples
		self.pool_size = pool_size
		self.strides = strides
		# border mode is either 'valid' or 'same'
		self.padding = padding

		# if border mode is same, use offsets to correct computed pooling regions (simulates padding)
		# initialize offset to 0, as it is not needed when border mode is valid
		self.row_offset = 0
		self.col_offset = 0
		if self.padding == 'same':

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
				self.row_offset = ((self.pool_output.shape[self.row_dim] - 1) * self.strides[0] +
								   self.pool_size[0] - self.pool_input.shape[self.row_dim]) // 2
				self.col_offset = ((self.pool_output.shape[self.col_dim] - 1) * self.strides[1] +
								   self.pool_size[1] - self.pool_input.shape[self.col_dim]) // 2

				# TODO: find alternative to these negative checks, seems to be produced when total stride length == length, and strides > pool size
				# computed offset can be negative, but this equals no offset
				if self.row_offset < 0:
					self.row_offset = 0
				if self.col_offset < 0:
					self.col_offset = 0
			else:
				# when using theano, padding is always added, and a pooling region center is never in the padding.
				# if there is no obvious center in the pooling region, unlike in 3x3, the max pooling
				# will use lower- and rightmost entry in the pooling region as the actual center, e.g. entry [3, 3] in a 4x4
				# region. if only the rows have an unclear center, the center is chosen to be the at the lowermost row and
				# the natural column center, e.g. entry [3, 1] in a 4x3 region. similarly, if only the columns have an
				# unclear center, the center is chosen to be at the natural row center and the rightmost column, e.g. [3, 1]
				# in a 3X4 region.

				# set offset (to simulate padding) to lowermost and rightmost entries by default
				self.row_offset = self.pool_size[0] - 1
				self.col_offset = self.pool_size[1] - 1

				# if rows have a clear center, update offset
				if self.pool_size[0] % 2 == 1:
					self.row_offset //= 2

				# if columns have a clear center, update offset
				if self.pool_size[1] % 2 == 1:
					self.col_offset //= 2

		super(MaxUnpooling2D, self).__init__(**kwargs)

	def _pooling_function(self, inputs, pool_size, strides, padding, data_format):

		indices = []

		# TODO: sample_no in potentially always 0, as deconv. model only ever receives one input image (images are switch specific in unpool, so >1 image for input makes no sense)
		# for every sample
		for sample_no in range(self.pool_output.shape[0]):
			# for every element in pooling output
			for i in range(self.pool_output.shape[self.row_dim]):
				# compute pooling region row indices
				start_row, end_row = self.compute_index_interval(i, 0, self.row_offset, self.row_dim)

				for j in range(self.pool_output.shape[self.col_dim]):
					# compute pooling region column indices
					start_col, end_col = self.compute_index_interval(j, 1, self.col_offset, self.col_dim)

					indices.extend(
						[self.get_region_max_index(sample_no, start_row, end_row, start_col, end_col, i, j, channel)
						 for channel in range(self.pool_output.shape[self.ch_dim])])

		if K.backend() == 'tensorflow':
			# use tensorflow
			recreated_input = tf.scatter_nd(indices=indices,
											updates=tf.reshape(inputs, [-1]),
											# updates=[recreated_output[i] for _, i in indices],
											shape=self.pool_input.shape,
											name=self.name + '_output')
		else:
			# use theano
			# very inefficient
			recreated_input = tht.zeros(self.pool_input.shape)
			for (input_index, output_index) in indices:
				recreated_input = tht.set_subtensor(recreated_input[input_index], inputs[output_index])

		return recreated_input

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
	def compute_index_interval(self, region_index, tuple_no, offset, dim):
		start_index = region_index * self.strides[tuple_no] - offset
		end_index = start_index + self.pool_size[tuple_no]

		# we ignore padded parts, so if offset makes start index smaller than smallest index in original pooling input
		# or end index larger than largest index in original pooling input, correct
		if start_index < 0:
			start_index = 0
		if end_index > self.pool_input.shape[dim]:
			end_index = self.pool_input.shape[dim]

		return start_index, end_index


deconv_example()
