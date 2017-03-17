from time import time
from os import mkdir, listdir
from os.path import join, dirname

import numpy as np

from keras.models import Model, load_model
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, Dropout, Flatten
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical

# import callbacks for visualizing
from visualizer.callbacks import AccuracyListSaver, LossListSaver, ActivationTupleListSaver

# find path to save networks and results
save_path = dirname(__file__)
networks_path = join(save_path, 'networks')
results_path = join(save_path, 'results')


def create_model(with_save=True):
	# define model
	inputs = Input(shape=(28, 28, 1))
	x = Convolution2D(32, 3, 3, activation='relu', border_mode='valid')(inputs)
	x = Convolution2D(32, 3, 3, activation='relu', border_mode='valid')(x)
	x = MaxPooling2D(pool_size=(2, 2))(x)
	x = Dropout(0.25)(x)
	x = Flatten()(x)
	x = Dense(128, activation='relu')(x)
	x = Dropout(0.5)(x)
	x = Dense(64, activation='relu')(x)
	predictions = Dense(10, activation='softmax')(x)

	# create and compile model with chosen hyperparameters
	model = Model(input=inputs, output=predictions)
	model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])

	# save model
	if with_save:
		save_to_disk(model)

	print('\nModel successfully created')

	return model


def train(model, no_of_epochs=10):
	# train top layers of model (self-defined layers)
	print('\n\nCommence MNIST model training\n')

	# initialize custom callbacks
	custom_callbacks = [AccuracyListSaver(results_path), LossListSaver(results_path), ActivationTupleListSaver(results_path)]

	# get data
	training_data, training_targets, test_data, test_targets = load_data()

	# train with chosen hyperparameters
	model.fit(training_data, training_targets, nb_epoch=no_of_epochs, batch_size=128, shuffle=True,
			  verbose=1, callbacks=custom_callbacks, validation_data=(test_data, test_targets))

	# save after training
	save_to_disk(model)

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
		classification = model.predict(test_data[i].reshape(1, 28, 28, 1))

		# find correct classification
		correct = test_targets[i]

		# count correctly classified, and print if incorrect
		if np.argmax(classification) == np.argmax(correct):
			correctly_classified += 1
		elif verbose:
			print('Incorrectly classified %d as %d' % (np.argmax(correct), np.argmax(classification)))

	print('Model correctly classified %d/%d MNIST images' % (correctly_classified, no_of_tests))


def load_data():
	# load
	(training_data, training_targets), (test_data, test_targets) = mnist.load_data()

	# reshape data
	training_data = training_data.reshape(training_data.shape[0], 28, 28, 1)
	test_data = test_data.reshape(test_data.shape[0], 28, 28, 1)

	# normalize data
	training_data = training_data.astype('float32')
	test_data = test_data.astype('float32')
	training_data /= 255.0
	test_data /= 255.0

	# make targets categorical
	training_targets = to_categorical(training_targets, 10)
	test_targets = to_categorical(test_targets, 10)

	return training_data, training_targets, test_data, test_targets


def save_to_disk(model):
	# make path, if not exists
	try:
		mkdir(networks_path)
		print('networks-folder created')
	except FileExistsError:
		# file exists, which is want we want
		pass

	model.save('%s/mnist_model.h5' % networks_path)

	print('\nModel saved as mnist_model.h5')


def load_from_disk():
	return load_model('%s/mnist_model.h5' % networks_path)


def main():
	# create a model, then train and test it.

	start_time = time()

	model = create_model()

	model = train(model)

	test(model, 1000, True)

	print('This took %.2f seconds' % (time() - start_time))


main()
