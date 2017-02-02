import numpy as np

import keras.backend as K
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D


# må finne måte å implementere max switch på
# kan potensielt bruke samme stil som for å hente alle activations, med K-funksjon


def simple_example():
	shape = (4, 4, 1)
	
	img = np.random.randint(0, 10, (1,) + shape)
	
	model = Sequential()
	model.add(MaxPooling2D((2, 2), input_shape=shape))
	
	model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
	
	pred = model.predict(img)
	
	print('Original:')
	array_print(img)
	
	print('Processed:')
	array_print(pred)


def adv_example():
	shape = (8, 8, 1)
	
	img = np.random.randint(0, 10, (1,) + shape)
	
	model = Sequential()
	model.add(Convolution2D(1, 2, 2, input_shape=shape))
	model.add(MaxPooling2D((3, 3), strides=(2, 2)))
	
	model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
	
	pred = model.predict(img)
	
	print('Original:')
	array_print(img)
	
	print('Processed:')
	print(pred)

	get_activation_tensor = K.function([model.input, K.learning_phase()], [model.layers[0].output])
	temp = get_activation_tensor([img, 0])[0]
	print('\n', model.layers[0].name)
	print('Shape:', temp.shape)
	array_print(temp)
	
	# print('Pure:', temp)
	
	get_activation_tensor = K.function([model.layers[1].input, K.learning_phase()], [model.layers[1].output])
	temp2 = get_activation_tensor([temp, 0])[0]
	print('\n', model.layers[1].name)
	print('Shape:', temp2.shape)
	array_print(temp2)
	
	# print(model.layers[1].get_config())
	# print(model.layers[1].pool_size)
	
	runs = 1
	test_speed = True
	if test_speed:
		runs = 1000000
	
	for _ in range(runs):
		unpooled_alt1 = find_max_pos(temp, model.layers[1].pool_size)
	# print('\nUnpooled alternative 1:')
	# array_print(unpooled_alt1)

	for _ in range(runs):
		unpooled_alt2 = unpool_with_mask(temp, temp2, model.layers[1].get_config())
	print('\nUnpooled alternative 2:')
	array_print(unpooled_alt2)
	
	
def get_submask(i, j, pool_input, pool_output, stride_width, stride_height, pool_width, pool_height):
	
	pool_area = pool_input[:, i * stride_width:i * stride_width + pool_width,
				j * stride_height:j * stride_height + pool_height, :]
	sub_mask = pool_output[:, i, j, :] == pool_area
	
	return pool_area, sub_mask


def get_true_mask(output_mask, i, j, sub_mask, pool_area, stride_width, stride_height, pool_width, pool_height):
	output_mask[:, i * stride_width:i * stride_width + pool_width, j * stride_height:j * stride_height + pool_height,:] \
		= get_log_or(output_mask, sub_mask * pool_area, i, j, stride_width, stride_height, pool_width, pool_height)


def get_log_or(output_mask, mask, i, j, stride_width, stride_height, pool_width, pool_height):
	return np.logical_or(output_mask[:, i * stride_width:i * stride_width + pool_width,
				  j * stride_height:j * stride_height + pool_height, :], mask)


# TODO: add support for channels, strides and border modes
# TODO: if valid, add outer zero (or -math.inf) padding, then remove?
def unpool_with_mask(pool_input, pool_output, pool_config):
	
	# print(pool_config)
	
	pool_height = pool_config['pool_size'][0]
	pool_width = pool_config['pool_size'][1]
	stride_height = pool_config['strides'][0]
	stride_width = pool_config['strides'][1]
	
	output_mask = np.empty(pool_input.shape)
	output_mask[:] = False
	
	for i in range(pool_output.shape[1]):
		for j in range(pool_output.shape[2]):
			start_x = i*stride_width
			end_x = start_x + pool_width
			start_y = j*stride_height
			end_y = start_y + pool_height
			
			pool_area = pool_input[:, start_x:end_x, start_y:end_y, :]
			sub_mask = pool_output[:, i, j, :] == pool_area
			# print('\nSubmask for', pool_output[:, i, j, :])
			# array_print(sub_mask)
			
			# TODO: check if strides are larger or smaller than pool size
			# TODO: on smaller strides, second update overwrites first if second has another max (maybe not anymore)
			output_mask[:, start_x:end_x, start_y:end_y, :] = np.logical_or(output_mask[:, start_x:end_x, start_y:end_y, :], sub_mask * pool_area)
			# print('\nUpdated output:')
			# array_print(output_mask)
	
	
	# unpool_mask = np.repeat(np.repeat(pool_output, pool_size[0], axis=1), pool_size[1], axis=2)
	
	# unpool_mask = pool_output
	# for i in range(len(pool_config['pool_size'])):
	# 	unpool_mask = np.repeat(unpool_mask, pool_config['pool_size'][i], axis=i+1)
	#
	# print('\nUnpool mask after step 1:')
	# array_print(unpool_mask)
	
	# if valid pooling is used, some of the outermost pixel might have been lost in pooling
	# if so, add None's to match pooling input shape
	# print('Shapes:', pool_input.shape, unpool_mask.shape)
	# axis = 1
	# while pool_input.shape != unpool_mask.shape:
		# unpool_mask = np.insert(unpool_mask, unpool_mask.shape[axis], None, axis=axis)
		# print('\nUpdated shapes:', pool_input.shape, unpool_mask.shape)
		# array_print(unpool_mask)
		# if axis == 1:
		# 	axis = 2
		# else:
		# 	axis = 1
	
	# if pool_input.shape != unpool_mask.shape:
	# 	output[:, :unpool_mask.shape[1], :unpool_mask.shape[2], :] = unpool_mask
	# 	unpool_mask = output
	
	# unpool_mask = unpool_mask == pool_input
	# print('\nUnpool mask after step 3:', unpool_mask)
	# array_print(unpool_mask)
	
	# return pool_input * unpool_mask
	return pool_input * output_mask
	
	
def find_max_pos(input_tensor, pool_size):
	
	# print('\nINPUT INFO:')
	# print(len(input_tensor))
	# print(len(input_tensor[0]))
	# print(len(input_tensor[0, 0]))
	# print(len(input_tensor[0, 0, 0]))
	#
	# print(pool_size[0])
	# print(pool_size[1])
	
	unpooled = np.zeros(input_tensor.shape)
	
	# print('\nMax. pos.:')
	for i in range(0, len(input_tensor[0]), pool_size[0]):
		for j in range(0, len(input_tensor[0, 0]), pool_size[1]):
			if i+pool_size[0] <= len(input_tensor[0, 1]) and j+pool_size[1] <= len(input_tensor[0, 2]):
				
				quadrant = input_tensor[0, i:i+pool_size[0], j:j+pool_size[1], 0]
				quad_max_pos = np.array(np.unravel_index(np.argmax(quadrant), quadrant.shape))
				full_max_pos = quad_max_pos + [i, j]
				
				# print('Make quadrant from [%d:%d] to [%d:%d]' % (i, i+pool_size[0], j, j+pool_size[1]))
				# print('Equivalent to quadrant staring at (%d,%d) and ending at (%d,%d)' % (i, j, i+pool_size[0]-1, j+pool_size[1]-1))
				# print(quadrant)
				# print('Quad. max.:', np.amax(quadrant))
				# print('Quad. max. pos.:', np.unravel_index(np.argmax(quadrant), quadrant.shape))
				# print('Equivalent full max. pos.:', max_pos)
				# print('i:', i, ', j:', j, '\n')
				
				unpooled[0, full_max_pos[0], full_max_pos[1], 0] = quadrant[quad_max_pos[0], quad_max_pos[1]]
				
	return unpooled


def array_print(np_array):
	for i in range(len(np_array[0])):
		row = ''
		for j in range(len(np_array[0, i])):
			row += ' ' + str(np_array[0, i, j, 0])
		print(row)


def main():
	print('SIMPLE')
	simple_example()
	
	print('\n\nADVANCED')
	adv_example()

main()
	
	


# class Unpooling2D(Layer):
#     def __init__(self, poolsize=(2, 2), ignore_border=True):
#         super(Unpooling2D,self).__init__()
#         self.input = T.tensor4()
#         self.poolsize = poolsize
#         self.ignore_border = ignore_border
#
#     def get_output(self, train):
#         X = self.get_input(train)
#         s1 = self.poolsize[0]
#         s2 = self.poolsize[1]
#         output = X.repeat(s1, axis=2).repeat(s2, axis=3)
#         return output
#
#     def get_config(self):
#         return {"name":self.__class__.__name__,
#             "poolsize":self.poolsize,
#             "ignore_border":self.ignore_border}
