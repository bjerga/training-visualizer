from keras.engine import Model
from keras.layers import Flatten, Dense
from keras_vggface.vggface import VGGFace
from keras.preprocessing.image import img_to_array
import pickle
import math
import os
import random
import numpy as np
from PIL import Image

base_path = '/Users/annieaa/Documents/NTNU/Fordypningsprosjekt'

with open(os.path.join(base_path, 'imfdb_training_data.pickle'), 'rb') as f:
	training_data = pickle.load(f)


# custom parameters
nb_class = 98
hidden_dim = 1024

batch_size = 64
steps_per_epoch = math.ceil(len(training_data) / batch_size)
no_of_epochs = 5


def create_model():

	vgg_model = VGGFace(include_top=False, input_shape=(130, 130, 3))

	for layer in vgg_model.layers:
		layer.trainable = False

	last_layer = vgg_model.get_layer('pool5').output
	x = Flatten(name='flatten')(last_layer)
	x = Dense(hidden_dim, activation='relu', name='fc1')(x)
	x = Dense(hidden_dim, activation='relu', name='fc2')(x)
	out = Dense(nb_class, activation='softmax', name='predictions')(x)

	custom_vgg_model = Model(vgg_model.input, out)

	for layer in custom_vgg_model.layers:
		print(layer.name)

	custom_vgg_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

	return custom_vgg_model


def train_model(model):

	# Still need to add callbacks

	model.fit_generator(generator=data_generation(), steps_per_epoch=steps_per_epoch, epochs=no_of_epochs, verbose=1)


def data_generation():

	while True:
		x_data = []
		y_data = []

		indices = random.sample(range(len(training_data)), batch_size)

		for i in indices:
			img_path, id_vector, expression_vector = training_data[i]
			img = Image.open(os.path.join(base_path, img_path))
			x_data.append(img_to_array(img))
			y_data.append(id_vector)

		yield np.array(x_data), np.array(y_data)


def main():
	model = create_model()
	#train_model(model)


main()
