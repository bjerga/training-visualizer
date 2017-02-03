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
	model.add(MaxPooling2D((3, 3), strides=(3, 3)))
	
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
	test_speed = False
	if test_speed:
		runs = 1000000

	for _ in range(runs):
		unpooled_alt2 = unpool_with_mask(temp, temp2, model.layers[1].pool_size, model.layers[1].strides)
	print('\nUnpooled alternative 2:')
	array_print(unpooled_alt2)


# TODO: add support for channels, and border modes
# TODO: if valid, add outer zero (or -math.inf) padding, then remove?
# generates a recreated pooling input from pooling output and pooling configuration
# the recreated input is zero, except from entries where the pooling output entries where originally chosen from,
# where the value is the same as the corresponding pooling output entry
def unpool_with_mask(pool_input, pool_output, pool_size, strides):
	
	# pool size and strides are both a (height, width)-tuple
	
	# create initial mask with all zero entries in pooling input shape
	unpool_mask = np.zeros(pool_input.shape)
	
	# for every element in pooling output
	for i in range(pool_output.shape[1]):
		# compute pooling filter height indices
		start_height = i * strides[0]
		end_height = start_height + pool_size[0]
		for j in range(pool_output.shape[2]):
			# compute pooling filter width indices
			start_width = j * strides[1]
			end_width = start_width + pool_size[1]
			
			# find which elements in the original pooling area that match the pooling output for that area
			output_matches = pool_output[:, i, j, :] == pool_input[:, start_height:end_height, start_width:end_width, :]
			# print('\nSubmask for', pool_output[:, i, j, :])
			# array_print(output_matches)
			
			# update corresponding area in unpooling mask with the output matches
			unpool_mask[:, start_height:end_height, start_width:end_width, :] += output_matches
			# print('\nUpdated output:')
			# array_print(unpool_mask)

	# generate True/False-mask from all entries that have matched at least once
	unpool_mask = unpool_mask > 0
	
	# apply mask to pooling input to generate the desired, recreated input
	return pool_input * unpool_mask


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
