import numpy as np

import keras.backend as K
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D


def simple_example():
	shape = (4, 4, 3)
	
	img = np.random.randint(0, 10, (1,) + shape)
	
	model = Sequential()
	model.add(MaxPooling2D((2, 2), input_shape=shape))
	
	pred = model.predict(img)
	
	print('Original:')
	array_print(img)
	
	print('Processed:')
	array_print(pred)
	
	print('Unpooled:')
	array_print(unpool_with_mask(img, pred, (2, 2)))


def adv_example():
	num = 7
	samples = 1
	filters = 2
	
	if K.image_dim_ordering() == 'tf':
		shape = (num, num, 3)
		max_pattern = np.zeros((samples, num-1, num-1, filters))
		copy_pattern = np.zeros((samples, num-1, num-1, filters))
	else:
		shape = (3, num, num)
		max_pattern = np.zeros((samples, filters, num-1, num-1))
		copy_pattern = np.zeros((samples, filters, num-1, num-1))
	
	for _ in range(100):
		img = np.random.randint(0, 10, (samples,) + shape)
		
		model = Sequential()
		model.add(Convolution2D(filters, 2, 2, input_shape=shape))
		model.add(MaxPooling2D((4, 4), strides=(5, 5), border_mode='same'))
	
		get_activation_tensor = K.function([model.input, K.learning_phase()], [model.layers[0].output])
		pool_input = get_activation_tensor([img, 0])[0]
		# print('\n', model.layers[0].name, '\nShape:', pool_input.shape)
		# array_print(pool_input)
		
		get_activation_tensor = K.function([model.layers[1].input, K.learning_phase()], [model.layers[1].output])
		pool_output = get_activation_tensor([pool_input, 0])[0]
		# print('\n', model.layers[1].name, '\nShape:', pool_output.shape)
		# array_print(pool_output)
	
		max_layer = model.layers[1]
		unpooled = unpool_with_mask(pool_input, pool_output, max_layer.pool_size, max_layer.strides, max_layer.border_mode)
		copy_pattern += unpooled == pool_input

		for s in range(pool_output.shape[0]):
			if K.image_dim_ordering() == 'tf':
				for i in range(pool_output.shape[1]):
					for j in range(pool_output.shape[2]):
						max_pattern += pool_output[s, i, j, :] == pool_input
			else:
				for i in range(pool_output.shape[2]):
					for j in range(pool_output.shape[3]):
						max_pattern += np.transpose(pool_output[s, :, i, j] == np.transpose(pool_input, axes=(0, 2, 3, 1)), axes=(0, 3, 1, 2))
				
	get_activation_tensor = K.function([model.input, K.learning_phase()], [model.layers[0].output])
	pool_input = get_activation_tensor([img, 0])[0]
	print('\n', model.layers[0].name, '\nShape:', pool_input.shape)
	array_print(pool_input)

	get_activation_tensor = K.function([model.layers[1].input, K.learning_phase()], [model.layers[1].output])
	pool_output = get_activation_tensor([pool_input, 0])[0]
	print('\n', model.layers[1].name, '\nShape:', pool_output.shape)
	array_print(pool_output)

	max_layer = model.layers[1]
	unpooled = unpool_with_mask(pool_input, pool_output, max_layer.pool_size, max_layer.strides, max_layer.border_mode)
	print('\nUnpooled:')
	array_print(unpooled)

	max_pattern = max_pattern > 0
	print('\nMax. output shape:', pool_output.shape)
	array_print(max_pattern)

	copy_pattern = copy_pattern > 0
	print('\nCopy:')
	array_print(copy_pattern)
	

# generates a recreated pooling input from pooling output and pooling configuration
# the recreated input is zero, except from entries where the pooling output entries where originally chosen from,
# where the value is the same as the corresponding pooling output entry
def unpool_with_mask(pool_input, pool_output, pool_size, strides=None, border_mode='valid'):
	
	# pool size and strides are both (rows, columns)-tuples
	# border mode is either 'valid' or 'same'
	
	# check backend used to detect dimensions used
	if K.image_dim_ordering() == 'tf':
		# tensorflow is used, dimension are (samples, rows, columns, feature maps/channels)
		row_dim = 1
		column_dim = 2
	else:
		# theano is used, dimension are (samples, feature maps/channels, rows, columns)
		row_dim = 2
		column_dim = 3
		
	# if strides are not specified, default to pool_size
	if strides is None:
		strides = pool_size
	
	# initialize offset to 0, as it is not needed when border mode is valid
	row_offset = 0
	column_offset = 0
	
	# if border mode is same, use offsets to correct computed pooling regions (simulates padding)
	if border_mode == 'same':
		
		if K.image_dim_ordering() == 'tf':
			# when using tensorflow, padding is not always added, and a pooling region center is never in the padding.
			# if there is no obvious center in the pooling region, unlike in 3x3, the max pooling will use upper- and
			# leftmost entry of the possible center entries as the actual center, e.g. entry [1, 1] in a 4x4 region.
			# padding will be added if it is possible to fit another center within the original tensor. padding will
			# also be added to balance the how much of each region is inside the original tensor, e.g. padding to get
			# two regions with 75% of entries inside each instead of one region with 100% inside and the other with 25%
			# inside
			
			# find offset (to simulate padding) by computing total region space that falls outside of original tensor
			# and divide by two to distribute to top-bottom/left-right of original tensor
			row_offset = (((pool_output.shape[row_dim] - 1) * strides[0] + pool_size[0]) - pool_input.shape[row_dim]) // 2
			column_offset = (((pool_output.shape[column_dim] - 1) * strides[1] + pool_size[1]) - pool_input.shape[column_dim]) // 2
			
			# TODO: find alternative to these negative checks, seems to be produced when total stride length == length, and strides > pool size
			# computed offset can be negative, but this equals no offset
			if row_offset < 0:
				row_offset = 0
			if column_offset < 0:
				column_offset = 0
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
			column_offset = pool_size[1] - 1
			
			# if rows have a clear center, update offset
			if pool_size[0] % 2 == 1:
				row_offset //= 2
			
			# if columns have a clear center, update offset
			if pool_size[1] % 2 == 1:
				column_offset //= 2
		
	# create initial mask with all zero (False) entries in pooling input shape
	unpool_mask = np.zeros(pool_input.shape)
	
	# for every sample
	for sample_no in range(pool_output.shape[0]):
		# for every element in pooling output
		for i in range(pool_output.shape[row_dim]):
			# compute pooling region row indices
			start_row = i * strides[0] - row_offset
			end_row = start_row + pool_size[0]
			
			# we ignore padded parts, so if offset makes start row negative, correct
			# no correction of end row is required, as list[start:end+positive num] is equivalent to list[start:end]
			if start_row < 0:
				start_row = 0
			
			for j in range(pool_output.shape[column_dim]):
				# compute pooling region column indices
				start_column = j * strides[1] - column_offset
				end_column = start_column + pool_size[1]
				
				# we ignore padded parts, as with rows
				if start_column < 0:
					start_column = 0
				
				if K.image_dim_ordering() == 'tf':
					# use tensorflow dimensions
					
					# find which elements in the original pooling area that match the pooling output for that area
					output_matches = pool_output[sample_no, i, j, :] == pool_input[sample_no, start_row:end_row, start_column:end_column, :]
					
					# update corresponding area in unpooling mask with the output matches
					unpool_mask[sample_no, start_row:end_row, start_column:end_column, :] += output_matches
				else:
					# use theano dimensions
					
					# as with tf, but transpose original pooling input to find values equal to max for all filters
					output_matches = pool_output[sample_no, :, i, j] == np.transpose(pool_input[sample_no, :, start_row:end_row, start_column:end_column], axes=(1, 2, 0))
					
					# as with tf, but transpose back to original form before addition
					unpool_mask[sample_no, :, start_row:end_row, start_column:end_column] += np.transpose(output_matches, axes=(2, 0, 1))
					
	# generate True/False-mask from all entries that have matched at least once
	unpool_mask = unpool_mask > 0
	
	# apply mask to pooling input to generate the desired, recreated input
	return pool_input * unpool_mask


def array_print(np_array):
	if K.image_dim_ordering() == 'tf':
		# use tensorflow dimensions
		row_count = np_array.shape[1]
		column_count = np_array.shape[2]
		filter_count = np_array.shape[3]
	else:
		# use theano dimensions
		row_count = np_array.shape[2]
		column_count = np_array.shape[3]
		filter_count = np_array.shape[1]
	
	for s in range(np_array.shape[0]):
		print('\n\nFor sample no. %d:' % s)
		for f in range(filter_count):
			print('\nFor feature map %d:' % f)
			for i in range(row_count):
				print_row = ''
				for j in range(column_count):
					if K.image_dim_ordering() == 'tf':
						print_row += ' ' + '%.5f' % np_array[s, i, j, f:f+1][0]
					else:
						print_row += ' ' + '%.5f' % np_array[s, f:f+1, i, j][0]
				print(print_row)


def main():
	# print('SIMPLE')
	# simple_example()

	print('\n\nADVANCED')
	adv_example()

main()
