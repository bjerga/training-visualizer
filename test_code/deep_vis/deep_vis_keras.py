import numpy as np

from scipy.misc import toimage
from scipy.ndimage.filters import gaussian_filter
from os import mkdir
from os.path import dirname, join
from time import time

from keras.models import Model
from keras.layers import Dense
from keras import backend as K
from keras.applications.vgg16 import VGG16

# define output path and make folder
output_path = join(dirname(__file__), 'output')
try:
	mkdir(output_path)
except FileExistsError:
	# folder exists, which is what we wanted
	pass

# set channel dimension based on image data format from Keras backend
if K.image_data_format() == 'channels_last':
	ch_dim = 3
else:
	ch_dim = 1

# for VGG16 specific testing
is_VGG16 = True
VGG16_MEAN_VALUES = np.array([103.939, 116.779, 123.68])

# set learning rate
learning_rate = 2500.0

# how many times we update image
no_of_iterations = 500

# specify L2-decay
# used to prevent a small number of extreme pixel values from dominating the output image
l2_decay = 0.0001

# specify frequency of blurring and standard deviation for kernel for Gaussian blur
# used to penalize high frequency information in the output image
blur_interval = 4
# standard deviation values between 0.0 and 0.3 work poorly, according to yosinski
blur_std = 1.0

# specify value percentile limit
# used to induce sparsity by setting pixels with small absolute value to zero
value_percentile = 0

# specify norm percentile limit
# used to induce sparsity by setting pixels with small norm to zero
norm_percentile = 0

# specify contribution percentile limit
# used to induce sparsity by setting pixels with small contribution to zero
contribution_percentile = 0

# specify absolute contribution percentile limit
# used to induce sparsity by setting pixels with small absolute contribution to zero
abs_contribution_percentile = 0

# choose whether to include regularization
regularize = True


# utility function used to convert an array into a savable image array
def deprocess(vis_array):

	# remove batch dimension, and alter color dimension accordingly
	img_array = vis_array[0]

	if K.image_data_format() == 'channels_first':
		# alter dimensions from (color, height, width) to (height, width, color)
		img_array = img_array.transpose((1, 2, 0))

	if is_VGG16:
		# add mean values
		img_array += VGG16_MEAN_VALUES.reshape((1, 1, 3))
		
		# change back to RGB
		img_array = img_array[:, :, ::-1]

	# clip in [0, 255], and convert to uint8
	img_array = np.clip(img_array, 0, 255).astype('uint8')
	
	return img_array


# creates a model to generate gradients from
def create_model():
	base_model = VGG16(include_top=True, weights='imagenet')

	# save weights from last layer (softmax)
	softmax_weights = base_model.layers[-1].get_weights()

	# create new last layer for model with linear activation and connect to same layer as old layer
	out = Dense(1000, activation='linear', weights=softmax_weights)(base_model.layers[-1].input)

	return Model(base_model.input, out)


# saves the visualization and a text file describing its creation environment
def save_visualization(img, layer_no, unit_index, loss_value):
	
	# create appropriate name to identify image
	if regularize:
		img_name = 'regularized'
	else:
		img_name = 'vanilla'
	img_name += '_{}_{}_{}'.format(layer_no, unit_index, time())

	# save the resulting image to disk
	# avoid scipy.misc.imsave because it will normalize the image pixel value between 0 and 255
	toimage(img).save(join(output_path, img_name + '.png'))

	# also save a txt-file containing information about creation environment and obtained loss
	img_info = 'Image "{}.png" was created from unit {} in layer {}, using the following hyperparameters:\n\n' \
			   'Learning rate: {}\n' \
			   'Number of iterations: {}\n' \
			   '----------\n' \
			   ''.format(img_name, unit_index, layer_no, learning_rate, no_of_iterations)
	if regularize:
		img_info += 'Regularization enabled\n\n' \
					'L2-decay: {}\n' \
					'Blur interval and std: {} & {}\n' \
					'Value percentile: {}\n' \
					'Norm percentile: {}\n' \
					'Contribution percentile: {}\n' \
					'Abs. contribution percentile: {}\n' \
					''.format(l2_decay, blur_interval, blur_std, value_percentile, norm_percentile,
							  contribution_percentile, abs_contribution_percentile)
	else:
		img_info += 'Regularization disabled\n'
	img_info += '----------\n' \
				'Obtained loss value: {}\n' \
				''.format(loss_value)
	with open(join(output_path, img_name + '_info.txt'), 'w') as f:
		f.write(img_info)
		
	print('\nImage of unit {} from layer {} have been saved as {}.png\n'.format(unit_index, layer_no, img_name))


# returns a function for computing loss and gradients w.r.t. the activations for the chosen unit in the output tensor
def get_loss_and_gradient_function(input_tensor, output_tensor, unit_index):
	
	# if unit index is specified as integer, convert to tuple
	if isinstance(unit_index, int):
		unit_index = (unit_index,)
		
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
	gradients = K.gradients(loss, input_tensor)[0]
	
	# return function returning the loss and gradients given a visualization image
	# add a flag to disable the learning phase
	return K.function([input_tensor, K.learning_phase()], [loss, gradients])


# creates an random, initial image to manipulate into a visualization
def create_initial_image(model_input_shape):
	
	# TODO: remove when done with testing
	# set random seed to be able to reproduce initial state of image
	# used in testing only, and should be remove upon implementation with tool
	np.random.seed(1337)
	
	# add (1,) for batch dimension
	return np.random.normal(0, 10, (1,) + model_input_shape[1:])


# regularizes visualization with various techniques
# each technique is activated by non-zero values for their respective global variables
def apply_ensemble_regularization(visualization, pixel_gradients, iteration_no):
	
	# regularizer #1
	# apply L2-decay
	if l2_decay > 0:
		visualization *= (1 - l2_decay)

	# regularizer #2
	# apply Gaussian blur
	if blur_interval > 0 and blur_std > 0:
		# only blur at certain iterations, as blurring is expensive
		if not iteration_no % blur_interval:
			# define standard deviations for blur kernel
			blur_kernel_std = [0, blur_std, blur_std, blur_std]
			
			# blur along height and width, but not along channel (color) dimension
			blur_kernel_std[ch_dim] = 0
			
			# perform blurring
			visualization = gaussian_filter(visualization, sigma=blur_kernel_std)

	# regularizer #3
	# apply value limit
	if value_percentile > 0:
		# find absolute values
		abs_visualization = abs(visualization)
		
		# find mask of high values (values above chosen value percentile)
		high_value_mask = abs_visualization >= np.percentile(abs_visualization, value_percentile)
		
		# apply to image to set pixels with small values to zero
		visualization *= high_value_mask

	# regularizer #4
	# apply norm limit
	if norm_percentile > 0:
		# compute pixel norms along channel (color) dimension
		pixel_norms = np.linalg.norm(visualization, axis=ch_dim)
		
		# find initial mask of high norms (norms above chosen norm percentile)
		high_norm_mask = pixel_norms >= np.percentile(pixel_norms, norm_percentile)
		
		# expand mask to account for color dimension
		high_norm_mask = expand_for_color(high_norm_mask)

		# apply to image to set pixels with small norms to zero
		visualization *= high_norm_mask

	# regularizer #5
	# apply contribution limit
	if contribution_percentile > 0:
		# predict the contribution of each pixel
		predicted_contribution = -visualization * pixel_gradients
	
		# sum over channel (color) dimension
		contribution = predicted_contribution.sum(ch_dim)

		# find initial mask of high contributions (contr. above chosen contr. percentile)
		high_contribution_mask = contribution >= np.percentile(contribution, contribution_percentile)

		# expand mask to account for color dimension
		high_contribution_mask = expand_for_color(high_contribution_mask)

		# apply to image to set pixels with small contributions to zero
		visualization *= high_contribution_mask

	# regularizer #6
	# apply absolute contribution limit
	if abs_contribution_percentile > 0:
	
		# alternative approach
		# predict the contribution of each pixel
		predicted_contribution = -visualization * pixel_gradients
	
		# sum over channel (color) dimension, and find absolute value
		abs_contribution = abs(predicted_contribution.sum(ch_dim))
		
		# find initial mask of high absolute contributions (abs. contr. above chosen abs. contr. percentile)
		high_abs_contribution_mask = abs_contribution >= np.percentile(abs_contribution, abs_contribution_percentile)

		# expand mask to account for color dimension
		high_abs_contribution_mask = expand_for_color(high_abs_contribution_mask)

		# apply to image to set pixels with small absolute contributions to zero
		visualization *= high_abs_contribution_mask

	return visualization


# use to expand a (batch, height, width)-numpy array with a channel (color) dimension
def expand_for_color(np_array):
	
	# expand at channel (color) dimension
	np_array = np.expand_dims(np_array, axis=ch_dim)
	
	# create tile repetition list, repeating thrice in channel (color) dimension
	tile_reps = [1, 1, 1, 1]
	tile_reps[ch_dim] = 3

	# apply tile repetition
	np_array = np.tile(np_array, tile_reps)
	
	return np_array


def main():
	# create model to generate gradients from
	model = create_model()

	# select units to visualize for by adding (layer number, unit index), where unit index is tuple for layers with
	# 3D structured output, like convolutional and pooling layers
	# units_to_visualize = [(22, 130), (2, 351), (22, 736), (22, 850)]
	# units_to_visualize = [(22, 402), (22, 587), (22, 950)]
	units_to_visualize = [(1, (112, 112, ch)) for ch in range(1)]
	# unit indices in last layer represent the following classes:
	# 130 flamingo, 351 hartebeest, 736 pool table, 850 teddy bear

	# for the chosen layer number and unit index
	for layer_no, unit_index in units_to_visualize:
		print('\nProcessing unit {} in layer {}'.format(unit_index, layer_no))
		
		# used to time generation of each image
		start_time = time()
		
		if layer_no < 0 or layer_no >= len(model.layers):
			raise ValueError('Invalid layer number {}: Layer numbers should be between {} and {}'.format(layer_no, 0, len(model.layers) - 1))
		
		# create and save loss and gradient function for current unit
		compute_loss_and_gradients = get_loss_and_gradient_function(model.input, model.layers[layer_no].output, unit_index)
	
		# create an initial visualization image
		visualization = create_initial_image(model.input_shape)
		
		# perform gradient ascent update with or without regularization for n steps
		for i in range(1, no_of_iterations + 1):
			
			# compute loss and gradient values (input 0 as arg. #2 to deactivate training layers, like dropout)
			loss_value, pixel_gradients = compute_loss_and_gradients([visualization, 0])
	
			# update visualization image
			visualization += pixel_gradients * learning_rate
	
			# if regularization has been activated, regularize image
			if regularize:
				visualization = apply_ensemble_regularization(visualization, pixel_gradients, i)
	
			# print('Current loss value:', loss_value)
			print('Round {} finished.'.format(i))

		# process visualization to match with standard image dimensions
		visualization_image = deprocess(visualization)
	
		# save visualization image, complete with info about creation environment
		save_visualization(visualization_image, layer_no, unit_index, loss_value)
		
		print('Visualization for unit {} from layer {} completed in {:.4f} seconds'.format(unit_index, layer_no, time() - start_time))

main()
