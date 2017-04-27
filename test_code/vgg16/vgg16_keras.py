import pickle
from time import time
from os import listdir
from os.path import join, dirname

import numpy as np

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image

# import callbacks for visualizing
from visualizer.callbacks import NetworkSaver, AccuracyListSaver, LossListSaver, ActivationTupleListSaver, SaliencyMaps, \
	DeepVisualization, Deconvolution

# find path to save networks and results
save_path = dirname(__file__)

# find path to imagenet URLs
imagenet_path = save_path.replace('vgg16', 'imagenet')
train_data_path = join(imagenet_path, 'train_data')
test_data_path = join(imagenet_path, 'test_data')
val_data_path = join(imagenet_path, 'val_data')

train_data_names = listdir(train_data_path)
train_data_amount = len(train_data_names)

with open(join(imagenet_path, 'wnid_index_map.pickle')) as f:
	wnid_index_map = pickle.load(f)


def create_model(input_shape):
	# define model
	model = VGG16(include_top=True, weights='imagenet', input_shape=input_shape)
	model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
	
	print('\nModel successfully created')
	
	return model


def train(model, no_of_epochs=10):
	# train top layers of model (self-defined layers)
	print('\n\nCommence VGG16 model training\n')
	
	# initialize custom callbacks
	custom_callbacks = [NetworkSaver(save_path), AccuracyListSaver(save_path), LossListSaver(save_path),
						SaliencyMaps(save_path),
						DeepVisualization(save_path,
										  [(-1, 0), (-1, 1), (-1, 2), (-1, 3), (-1, 4), (-1, 5), (-1, 6), (-1, 7),
										   (-1, 8), (-1, 9)],
										  2500.0, 500, l2_decay=0.0001, blur_interval=4, blur_std=1.0),
						Deconvolution(save_path, feat_map_layer_no=3, feat_map_amount=3)]
	
	batch_size = 256
	steps_per_epoch = train_data_amount / batch_size
	
	print('Steps per epoch:', steps_per_epoch)
	
	my_gen = data_generator()
	
	
	# train with chosen hyperparameters
	model.fit_generator(generator=my_gen, steps_per_epoch=steps_per_epoch, epochs=30, verbose=1, )
	
	print('\nCompleted VGG16 model training\n')
	
	return model

def data_generator(batch_size):
	
	while True:
		random_img_names = np.random.choice(train_data_names, batch_size, replace=False)
		
		x_data = [preprocess_data(load_image(train_data_path, img_name)) for img_name in random_img_names]
		y_data = [wnid_index_map.index(img_name) for img_name in random_img_names]
		
		yield x_data, y_data

def load_image(img_path, img_name):
	img = image.load_img(join(img_path, img_name))
	return image.img_to_array(img)

def preprocess_data(img_array):
	return img_array


def postprocess_data(img_array):
	return img_array


def main():
	# create a model, then train and test it.
	
	start_time = time()
	
	model = create_model()
	
	model = train(model)
	
	print('This took %.2f seconds' % (time() - start_time))


main()
