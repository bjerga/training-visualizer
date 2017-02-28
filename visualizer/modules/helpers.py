import pickle
import subprocess as sub
from time import time, sleep
from os import listdir, mkdir, remove
from os.path import join, getmtime
from shutil import move

from flask_login import current_user

import numpy as np
from PIL import Image

from visualizer.modules.models import User, Tag, FileMeta


# allowed extension for upload files
ALLOWED_EXTENSIONS = {'py'}


# get current user in form of string
def get_current_user():
	return str(current_user)


# check is username is unique in database
def unique_username(username):
	if User.query.filter_by(username=username).first():
		return False
	return True


# to be implemented
def has_permission(next_access):
	# TODO: implement
	return True


# check if filename is unique for current user in database
def unique_filename(filename):
	if FileMeta.query.filter_by(filename=filename, owner=get_current_user()).first():
		return False
	return True


# check if file extension is allowed
def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


# TODO: might be possible to manage this directly in the query
# check if tag already exists
# if yes, return existing tag
# if no, return new tag
def get_existing_tag(text):
	existing_tag = Tag.query.filter_by(text=text).first()
	if existing_tag:
		return existing_tag
	return Tag(text)
		

# get all errors detected in the form, and return in readable format
def get_form_errors(form):
	form_errors = []
	for field, errors in form.errors.items():
		for error in errors:
			form_errors.append('%s - %s' % (getattr(form, field).label.text, error))
	return form_errors


# check if a network has any files associated with it, i.e. networks or visualization data
def has_associated_files(file_folder):
	network_folder = join(file_folder, 'networks')
	result_folder = join(file_folder, 'results')
	return listdir(network_folder) or listdir(result_folder)


# run a python program via command line
def run_python_shell(file_path, shared_bool):
	if file_path:
		
		print('\nSubprocess started\n')
		
		# run program via command line
		sub.run('python3 ' + file_path, shell=True)

		print('\nSubprocess finished\n')
	else:
		print('\n\nNo file found\n\n')
	
	# mark via shared boolean that process is no longer writing
	shared_bool.value = False
	
	# make sure process ends (not certain this is needed)
	return


# TODO: currently only works for black and white images, must also work for RGB
# create activation images for all layers
def visualize_activations(pickle_file, results_path, activations_path):

	# remove old visualization files before creating new ones
	for old_file in listdir(activations_path):
		remove(join(activations_path, old_file))

	# need time of creation to ensure unique filename
	creation_time = time()

	# read content of pickle file
	with open(join(results_path, pickle_file), 'rb') as f:
		content_list = pickle.load(f)

	# for all layers
	for layer_no in range(len(content_list)):
		layer_name, layer_activation = content_list[layer_no]

		# print('\nLength of activation shape', layer_activation.shape, 'is', len(layer_activation.shape), '\n')

		# scale to fit between [0.0, 255.0]
		layer_activation += max(-np.min(layer_activation), 0.0)
		la_max = np.max(layer_activation)
		if la_max != 0.0:
			layer_activation /= la_max
		layer_activation *= 255.0

		# remove unnecessary outer list for everything but one-dimensional arrays
		# 1D arrays needs the extra dimension to be accepted by Image.fromarray(...)
		if len(layer_activation[0].shape) != 1:
			layer_activation = layer_activation[0]

		# if activation has no channels
		if len(layer_activation.shape) < 3:
			img = Image.fromarray(layer_activation.astype('uint8'), 'L')
			img.save(join(activations_path, 'ln%d_%s_%d.png' % (layer_no, layer_name, creation_time)))
		# if activation has channels (typically activations from convolution layers)
		else:
			for channel in range(layer_activation.shape[2]):
				img = Image.fromarray(layer_activation[:, :, channel].astype('uint8'), 'L')
				img.save(join(activations_path, 'ln%d_%s_ch%d_%d.png' % (layer_no, layer_name, channel, creation_time)))

		# print('\n\n\n', layer_name, 'has shape', layer_activation.shape, '\n\n\n')


# create all folders in list at the specified base path
def create_folders(base_path, new_folders):
	for new_folder in new_folders:
		try:
			mkdir(join(base_path, new_folder))
		except FileExistsError:
			pass
