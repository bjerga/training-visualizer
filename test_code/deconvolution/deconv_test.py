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
from keras.layers import Input, Conv2DTranspose, Activation
from keras.layers.pooling import _Pooling2D
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image

# TODO: delete when done with testing
# for images from URLs
from PIL import Image
import requests
from io import BytesIO


# for alternate preprocessing:
alt = False

# VGG16 mean values
MEAN_VALUES = np.array([103.939, 116.779, 123.68])

# TODO: test with img-model map
model_map = {}


# TODO: delete when done with testing
# NOTE: throws RequestException, ValueError, OSError
def load_image_from_url(url):

	response = requests.get(url, timeout=5)

	img = Image.open(BytesIO(response.content))
	img = scipy.misc.imresize(image.img_to_array(img), (224, 224))
	img = img.astype('float64')

	return preprocess_image(img)


def load_from_file(img_name):
	img_path = join(dirname(__file__), 'input', img_name)
	img = image.load_img(img_path, target_size=(224, 224))
	img = image.img_to_array(img)

	return preprocess_image(img)


def preprocess_image(img):
	img -= MEAN_VALUES.reshape((1, 1, 3))
	
	if K.image_data_format() == 'channels_first':
		img = img.transpose((2, 0, 1))
	
	return np.expand_dims(img, axis=0)


# utility function used to convert a tensor into a savable image
def tensor_to_img(vis_tensor):

	# remove batch dimension
	img = vis_tensor[0]
	
	if K.image_data_format() == 'channels_first':
		# alter dimensions from (color, height, width) to (height, width, color)
		img = img.transpose((1, 2, 0))

	img += MEAN_VALUES.reshape((1, 1, 3))

	img = np.clip(img, 0, 255).astype('uint8')  # clip in [0;255] and convert to int
	
	return img


# saves the visualization and a txt-file describing its creation environment
def save_reconstruction(recon, feat_map_no, alt=False):
	
	# process before save
	img = tensor_to_img(recon)
	
	# define output path and make folder
	output_path = join(dirname(__file__), 'output')
	try:
		mkdir(output_path)
	except FileExistsError:
		# folder exists, which is what we wanted
		pass
	
	# TODO: delete alt after testing
	if alt:
		image_name = 'test_%d_feat_map_%d_alt' % (len(listdir(output_path)), feat_map_no)
	else:
		image_name = 'test_%d_feat_map_%d' % (len(listdir(output_path)), feat_map_no)
	# save the resulting image to disk
	scipy.misc.toimage(img, cmin=0, cmax=255).save(join(output_path, image_name + '.png'))
	# avoid scipy.misc.imsave because it will normalize the image pixel value between 0 and 255
	
	# print('\nImage has been saved as %s.png\n' % image_name)


# creates a model to generate gradients from
def create_conv_model(input_shape, include_top):
	return VGG16(include_top=include_top, weights='imagenet', input_shape=input_shape)


def create_deconv_model(conv_model, img, img_name):
	global model_map
	
	start_time = time()
	
	try:
		deconv_model, layer_map = model_map[img_name]
		# print('\nLoad deconv. model')
	except KeyError:
		
		# print('\nCreate deconv. model')
		
		# create layer map between conv. layers and deconv. layers
		layer_map = {}
		
		if K.image_data_format() == 'channels_last':
			filt_dim = 3
		else:
			filt_dim = 1
		
		# TODO: find efficient way to compute pooling input and output
		unpool_info = {}
		start_layer_no = 0
		start_input = img
		for layer_no in range(len(conv_model.layers)):
			if 'pool' in conv_model.layers[layer_no].name:
				pool_input, pool_output = compute_layer_input_and_output(conv_model, layer_no, start_layer_no, start_input)
				
				unpool_info[layer_no] = (pool_input, pool_output)
				
				start_layer_no = layer_no + 1
				start_input = pool_output
			
		# use output shape of conv. model as input shape
		dc_input_shape = conv_model.output_shape[1:]
		
		# add first layer with input shape
		dc_input = Input(shape=dc_input_shape)
		
		# add other layers
		dc_layer_count = 1
		x = dc_input
		for layer_no in range(len(conv_model.layers) - 1, -1, -1):
			layer = conv_model.layers[layer_no]
			if 'conv' in layer.name:
				x = Activation(layer.activation)(x)
				x = Conv2DTranspose(filters=layer.input_shape[filt_dim],
									kernel_size=layer.kernel_size,
									strides=layer.strides,
									padding=layer.padding,
									data_format=layer.data_format,
									dilation_rate=layer.dilation_rate,
									# weights=flip_weights(layer.get_weights()),
									weights=[layer.get_weights()[0]],
									use_bias=False)(x)
				# TODO: may need to remove the +1 (currently it makes the deconvolution skip RELU-layer on intermediate feature map reconstructions)
				layer_map[layer_no] = dc_layer_count + 1
				dc_layer_count += 2
			elif 'pool' in layer.name:
				pool_input, pool_output = unpool_info[layer_no]
				x = Unpooling2D(pool_input=pool_input,
								pool_output=pool_output,
								pool_size=layer.pool_size,
								strides=layer.strides,
								padding=layer.padding)(x)
				layer_map[layer_no] = dc_layer_count
				dc_layer_count += 1
			else:
				# print('\nFound layer in original model which is neither convolutional or pooling, with layer name: ' + layer.name)
				pass
	
		deconv_model = Model(inputs=dc_input, outputs=x)
		
		model_map[img_name] = (deconv_model, layer_map)
	
	# print('\nTime to create deconv. model was %.4f seconds' % (time() - start_time))
	
	return deconv_model, layer_map


# TODO: might not needed due to keras 2.0 changes
# flip weights to fit a deconvolution layer
def flip_weights(convolution_weights):
	# TODO: add and test flip and transpose for theano
	# weights have dimensions (rows, columns, channels, filters)
	# flip across rows (axis=0) and then columns (axis=1)
	transformed_weights = np.fliplr(np.flipud(convolution_weights[0]))
	
	# switch RGB channels at dim. 2 with filter channels at dim. 3
	# creates three deconv. filters, R, G, and B, to apply to the feature map output of the previous conv. filters
	# TODO: in keras 2.0, this gives an error
	# transformed_weights = np.transpose(transformed_weights, axes=(0, 1, 3, 2))
	
	# wrap in list and return
	return [transformed_weights]


def deconv_example():
	print_meta = True

	img = load_from_file('dog.jpg')

	conv_model = create_conv_model(img.shape[1:], False)

	# print('\nLayers in conv. model:')
	# for layer in conv_model.layers:
	# 	print(layer.name)
	
	# print conv info
	if print_meta:
		print('\n***CONVOLUTIONAL MODEL INFO***')
		print('Conv. input shape:', conv_model.input_shape)
		print('Conv. output shape:', conv_model.output_shape)
		
	deconv_model, layer_map = create_deconv_model(conv_model, img, 'dog.jpg')
	
	# print('\nLayers in deconv. model:')
	# for layer in deconv_model.layers:
	# 	print(layer.name)
	
	# print deconv info
	if print_meta:
		print('\n***DECONVOLUTIONAL MODEL INFO***')
		print('Deconv. input shape:', deconv_model.input_shape)
		print('Deconv. output shape:', deconv_model.output_shape)

	# note that layers are zero indexed
	#  TODO: test other (lower) layers
	feat_map_layer = 18

	print('\nReady for deconv. pred.')
	# TODO: check other feature maps
	# deconv_viz = deconv_model.predict(preprocess_prediction(conv_feat_maps, i))
	# img = load_and_process('car.jpg')
	# produce_reconstructions(deconv_model, conv_model, img, layer_map, pool_layers, feat_map_layer, True)
	

	np.random.seed(1337)
	feat_map_random_subset = np.random.randint(0, conv_model.layers[feat_map_layer].output_shape[3], 3)
	for feat_map_no in feat_map_random_subset:
		print('\n***** Reconstruct for feature map %d in layer %d *****' % (feat_map_no, feat_map_layer))
		start_time = time()
		max_images, urls = get_max_images(100, 5, conv_model, feat_map_layer, feat_map_no)
		for i in range(len(max_images)):
			max_img = max_images[i]
			max_img_name = urls[i]
			produce_reconstructions(deconv_model, conv_model, max_img, max_img_name, layer_map, feat_map_layer, feat_map_no, True)
		print('\nTime to perform reconstructions for feat map no. %d was %.4f seconds' % (feat_map_no, time() - start_time))
	

# TODO: currently produces several reconstructions based on N feature maps with the top N single activation strengths
def produce_reconstructions(deconv_model, conv_model, img, img_name, layer_map, feat_map_layer, feat_map_no, alter_unpools=False):
	
	# if specified, update all unpooling layers (used when image input has changed)
	if alter_unpools:
		
		# create new model with updated unpooling layers (image specific)
		deconv_model, layer_map = create_deconv_model(conv_model, img, img_name)
			
	# get conv. model output (feat maps) for desired feature map layer
	_, feat_maps = compute_layer_input_and_output(conv_model, feat_map_layer, 0, img)
	
	# print(np.array_equal(feat_maps, produce_feat_maps(conv_model, img, feat_map_layer)))
	
	# find the N feature maps with the N strongest single activations
	# TODO: maybe not use N feature maps, but rather N variation of one feature map, each with one of N strongest activations
	# max_feature_maps = get_max_feature_map_indices(feat_maps, 10)
	# print('\nMax feature maps are:', max_feature_maps)
	
	# for all max feature maps
	# for feat_map_no in max_feature_maps:
	# TODO: delete alt. when done with testing
	if not alt:
		# preprocess feature maps with regard to chosen feature map
		processed_feat_maps = preprocess_feat_maps(feat_maps, feat_map_no)
		
		# feed to deconv. model to produce reconstruction
		_, reconstruction = compute_layer_input_and_output(deconv_model, -1, layer_map[feat_map_layer], processed_feat_maps)
		
		# save reconstruction to designated folder
		save_reconstruction(reconstruction, feat_map_no)
	else:
		processed_feat_maps = ppp_alt(feat_maps, feat_map_no)
		_, reconstruction = compute_layer_input_and_output(deconv_model, -1, layer_map[feat_map_layer], processed_feat_maps)
		save_reconstruction(reconstruction, feat_map_no, True)

	
def compute_layer_input_and_output(model, end_layer_no, start_layer_no, start_input):
	input_func = K.function([model.layers[start_layer_no].input, K.learning_phase()],
							[model.layers[end_layer_no].input])
	layer_input = input_func([start_input, 0])[0]

	output_func = K.function([model.layers[end_layer_no].input, K.learning_phase()],
							 [model.layers[end_layer_no].output])
	layer_output = output_func([layer_input, 0])[0]
	
	return layer_input, layer_output

	
def produce_feat_maps(conv_model, img, end_layer=-1):
	
	func = K.function([conv_model.input, K.learning_phase()],
					  [conv_model.layers[end_layer].output])
	return func([img, 0])[0]


# TODO: delete when done with testing
def ppp_alt(pred, feat_map_no):
	processed_pred = np.zeros(pred.shape)
	processed_pred[:, :, :, feat_map_no] = pred[:, :, :, feat_map_no]
	return processed_pred


def preprocess_feat_maps(feat_maps, feat_map_no):

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

	# save max activation
	max_activation = feat_maps[max_activation_pos]

	# set all entries except max activation of chosen feature map to zero
	processed_feat_maps = np.zeros(feat_maps.shape)
	processed_feat_maps[max_activation_pos] = max_activation

	return processed_feat_maps


def get_max_feature_map_indices(pred, amount):

	# find feature maps with largest single element values

	if K.image_data_format() == 'channels_last':
		feat_map_maxes = np.array([np.amax(pred[:, :, :, feat_map_no]) for feat_map_no in range(pred.shape[3])])
		max_feat_maps = feat_map_maxes.argsort()[-amount:]
	else:
		# TODO: test for theano
		feat_map_maxes = np.array([np.amax(pred[:, feat_map_no, :, :]) for feat_map_no in range(pred.shape[1])])
		max_feat_maps = feat_map_maxes.argsort()[-amount:]
		pass

	return max_feat_maps


def get_max_images(check_amount, chose_amount, conv_model, feat_map_layer_no, feat_map_no):

	urls = []
	scores = []

	with open(join(dirname(__file__), 'input', 'fall11_urls.txt'), 'r') as f:
		for i in range(check_amount):
			url = f.readline().split('\t')[1].rstrip()

			print('Check image ' + str(i) + ' at URL: ' + url)

			try:
				img = load_image_from_url(url)
			except RequestException:
				print('Error with request')
				continue
			except ValueError:
				print('Image no longer exists')
				continue
			except OSError:
				print('Error in opening image')
				continue

			urls.append(url)

			_, pred = compute_layer_input_and_output(conv_model, feat_map_layer_no, 0, img)

			scores.append(np.amax(pred[:, :, :, feat_map_no]))

	scores = np.array(scores)
	chosen_urls = []
	chosen_images = []
	i = 0
	print('\nChosen image URLs:')
	for index in scores.argsort()[-chose_amount:]:
		print(urls[index])
		chosen_urls.append(urls[index])

		# if image was safely loaded the first time, assume safe to load now
		img = load_image_from_url(urls[index])
		chosen_images.append(img)

		# save max images for comparison
		img = img[0]
		img = np.clip(img, 0, 255).astype('uint8')  # clip in [0;255] and convert to int
		scipy.misc.toimage(img, cmin=0, cmax=255).save(join(dirname(__file__), 'max_images', 'feat_map_%d_image_%d.png' % (feat_map_no, i)))

		i += 1

	return chosen_images, chosen_urls


def array_print(np_array):
	if K.image_data_format() == 'channels_last':
		# use tensorflow dimensions
		row_dim = 1
		column_dim = 2
		filter_dim = 3
	else:
		# use theano dimensions
		row_dim = 2
		column_dim = 3
		filter_dim = 1
	
	for s in range(np_array.shape[0]):
		for f in range(np_array.shape[filter_dim]):
			print('\nFor sample-feature map %d-%d:' % (s, f))
			for i in range(np_array.shape[row_dim]):
				print_row = ''
				for j in range(np_array.shape[column_dim]):
					if K.image_data_format() == 'channels_last':
						print_row += ' ' + '%.5f' % np_array[s, i, j, f:f+1][0]
					else:
						print_row += ' ' + '%.5f' % np_array[s, f:f+1, i, j][0]
				print(print_row)


def main():
	
	print('\n\nDeconvolution example')
	deconv_example()
	
	# print('\n\nUnpooling example')
	# unpool_example()


# generates a recreated pooling input from pooling output and pooling configuration
# the recreated input is zero, except from entries where the pooling output entries where originally chosen from,
# where the value is the same as the corresponding pooling output entry
class Unpooling2D(_Pooling2D):
	
	#########################################################
	### these three initial methods are required by Layer ###
	#########################################################
	
	def __init__(self, pool_input, pool_output, pool_size, strides, padding, **kwargs):
		
		# check backend to detect dimensions used
		if K.image_data_format() == 'channels_last':
			# tensorflow is used, dimension are (samples, rows, columns, filters)
			self.row_dim = 1
			self.col_dim = 2
			self.filt_dim = 3
		else:
			# theano is used, dimension are (samples, filters, rows, columns)
			self.row_dim = 2
			self.col_dim = 3
			self.filt_dim = 1
		
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
		
		super(Unpooling2D, self).__init__(**kwargs)

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
					
					indices.extend([self.get_region_max_index(sample_no, start_row, end_row, start_col, end_col, i, j, filt)
									for filt in range(self.pool_output.shape[self.filt_dim])])
		
		if K.backend() == 'tensorflow':
			# use tensorflow
			# print(indices)
			# fix this. consider only returning the first match in get_indices
			# recreated_input = tf.scatter_nd(indices=[i for i, _ in indices],
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
	
	def get_region_max_index(self, sample_no, start_row, end_row, start_col, end_col, i, j, filt):
		
		if K.backend() == 'tensorflow':
			# use tensorflow dimensions
			for row in range(start_row, end_row):
				for col in range(start_col, end_col):
					if self.pool_input[sample_no, row, col, filt] == self.pool_output[sample_no, i, j, filt]:
						return (sample_no, row, col, filt)
		else:
			# use theano dimensions
			for row in range(start_row, end_row):
				for col in range(start_col, end_col):
					if self.pool_input[sample_no, filt, row, col] == self.pool_output[sample_no, filt, i, j]:
						return ((sample_no, filt, row, col), (sample_no, filt, i, j))
	
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

main()
