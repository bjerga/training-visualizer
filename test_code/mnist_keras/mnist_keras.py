from time import time
from os.path import dirname

import numpy as np

import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical

# import callbacks for visualizing
from custom_keras.callbacks import CustomCallbacks

# find path to save networks and results
save_path = dirname(__file__)

if K.image_data_format() == 'channels_last':
	# use tensorflow dimensions
	input_shape = (28, 28, 1)
else:
	# use theano dimensions
	input_shape = (1, 28, 28)


def create_model():
	# define model
	inputs = Input(shape=input_shape)
	x = Conv2D(32, (3, 3), activation='relu', padding='valid')(inputs)
	x = Conv2D(32, (3, 3), activation='relu', padding='valid')(x)
	x = MaxPooling2D(pool_size=(2, 2))(x)
	x = Dropout(0.25)(x)
	x = Flatten()(x)
	x = Dense(128, activation='relu')(x)
	x = Dropout(0.5)(x)
	x = Dense(64, activation='relu')(x)
	predictions = Dense(10, activation='softmax')(x)

	# create and compile model with chosen hyperparameters
	model = Model(inputs=inputs, outputs=predictions)
	model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])

	print('\nModel successfully created')

	return model


def train(model, no_of_epochs=10):
	print('\n\nCommence MNIST model training\n')
	
	# initialize custom callbacks
	callbacks = CustomCallbacks(save_path)
	callbacks.register_network_saver()
	callbacks.register_training_progress()
	callbacks.register_layer_activations()
	callbacks.register_saliency_maps()
	callbacks.register_deconvolution_network(3, 16, interval=10)
	callbacks.register_deep_visualization([(9, i) for i in range(model.layers[9].output_shape[1])],
										  200.0, 50, l2_decay=0.0001, blur_interval=4, blur_std=1.0, interval=10)

	# get data
	training_data, training_targets, test_data, test_targets = load_data()

	# train with chosen hyperparameters
	model.fit(training_data, training_targets, epochs=no_of_epochs, batch_size=128, shuffle=True, verbose=1,
			  callbacks=callbacks.get_list(), validation_data=(test_data, test_targets))

	print('\nCompleted MNIST model training\n')

	return model


def test(model, no_of_tests=1, verbose=True):
	# get validation data
	_, _, test_data, test_targets = load_data()

	# find indices of random images to test on
	random_indices = np.random.randint(len(test_data), size=no_of_tests)

	# test for all indices and count correctly classified
	correctly_classified = 0
	for i in random_indices:

		# get model classification
		classification = model.predict(test_data[i].reshape((1,) + input_shape))

		# find correct classification
		correct = test_targets[i]

		# count correctly classified, and print if incorrect
		if np.argmax(classification) == np.argmax(correct):
			correctly_classified += 1
		elif verbose:
			print('Incorrectly classified {} as {}'.format(np.argmax(correct), np.argmax(classification)))

	print('Model correctly classified {}/{} MNIST images'.format(correctly_classified, no_of_tests))


def load_data():
	# load
	(training_data, training_targets), (test_data, test_targets) = mnist.load_data()

	# reshape data
	training_data = training_data.reshape((training_data.shape[0],) + input_shape)
	test_data = test_data.reshape((test_data.shape[0],) + input_shape)

	# normalize data
	training_data = training_data.astype('float32')
	test_data = test_data.astype('float32')
	training_data /= 255.0
	test_data /= 255.0

	# make targets categorical
	training_targets = to_categorical(training_targets, 10)
	test_targets = to_categorical(test_targets, 10)

	return training_data, training_targets, test_data, test_targets


def main():
	# create a model, then train and test it.

	start_time = time()

	model = create_model()

	model = train(model)

	test(model, 1000, True)

	print('This took {:.2f} seconds'.format(time() - start_time))


main()
