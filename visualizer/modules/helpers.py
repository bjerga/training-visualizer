import re
import pickle
import subprocess as sub
import matplotlib.pyplot as plt
from time import time, sleep
from os import listdir, mkdir, remove
from os.path import join, getmtime
from shutil import move

import numpy as np
from flask import abort
from flask_login import current_user
from PIL import Image

from visualizer.modules.models import User, Tag, FileMeta


ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'py'}


def valid_username(username):
	# checks if password only contains letters (capitalized or not), numbers and underscores
	if re.match('^[\w]*$', username):
		return True
	return False


def valid_password(password):
	# checks if password only contains letters (capitalized or not) and numbers,
	# and that it contains at least 1 capitalized letter,
	# and that it contains at least 1 non-capitalized letter,
	# and that it contains at least 2 numbers
	# if re.match('^[\w]*$', password) and \
	# 				len(re.findall('[A-Z]', password)) > 0 and \
	# 				len(re.findall('[a-z]', password)) > 0 and \
	# 				len(re.findall('[0-9]', password)) > 1:
	if True:
			return True
	return False


def unique_username(username):
	if User.query.filter_by(username=username).first():
		return False
	return True


def has_permission(next_access):
	# TODO: implement
	return True


def unique_filename(filename):
	if FileMeta.query.filter_by(filename=filename).first():
		return False
	return True


def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


# TODO: authorization could be improved
def check_authorization(username):
	if username != str(current_user):
		abort(401)


# TODO: might be possible to manage this directly in the query
def get_existing_tag(text):
	existing_tag = Tag.query.filter_by(text=text).first()
	if existing_tag:
		return existing_tag
	return Tag(text)
		

def get_form_errors(form):
	form_errors = []
	for field, errors in form.errors.items():
		for error in errors:
			form_errors.append('%s - %s' % (getattr(form, field).label.text, error))
	return form_errors


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


def visualize_callback_output(file_path, filename, shared_bool):

	# create file path for results
	results_path = file_path.replace(filename, 'results')

	# for visualization plot
	plots_path = file_path.replace(filename, 'plots')
	plot_colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w']

	# for activation visualization
	activations_path = file_path.replace(filename, 'activations')

	modification_times = {}

	# if other process is writing
	while shared_bool.value:

		# for all visualization files in results folder
		plot_num = 0
		for vis_file in listdir(results_path):

			if vis_file not in modification_times.keys():
				modification_times[vis_file] = 0

			# check if visualization file has changed from last time
			if modification_times[vis_file] < getmtime(join(results_path, vis_file)):

				# update modification time for visualization file
				modification_times[vis_file] = getmtime(join(results_path, vis_file))

				if vis_file.endswith('.txt'):
					plot_content(vis_file, plot_colors[plot_num], results_path, plots_path)
					plot_num += 1
				elif vis_file.endswith('.pickle'):
					visualize_activations(vis_file, results_path, activations_path)
				else:
					# unknown file format, cannot visualize
					print('\nUnknown file format, no visualization for file')

		# sleep to keep from constantly visualizing
		sleep(3)

	print('\nVisualization ended\n')

	# make sure process ends (not certain this is needed)
	return


def plot_content(text_file, plot_color, results_path, plots_path):

	# get filename without '.txt'
	filename = text_file[:-4]

	# read content of text file
	with open(join(results_path, text_file), 'r') as f:
		content_list = [float(line) for line in f]

	# remove old plot of file before creating new one
	[remove(join(plots_path, plot_file)) for plot_file in listdir(plots_path) if plot_file.startswith(filename)]

	# create new plot
	# plt.figure(figsize=(20, 10))
	plt.plot(content_list, plot_color + '-')

	# set plot title and labels
	x_label, y_label, model_no = filename.split('_')
	plt.title(('%s Over %s For Model No. %s' % (y_label, x_label, model_no)).title())
	plt.xlabel(x_label.title())
	plt.ylabel(y_label.title())

	# set limits of x to be outermost points
	plt.xlim([0, len(content_list) - 1])

	# save new plot
	plt.savefig(join(plots_path, '%s_plot_%d.png' % (filename, time())))

	# clear and close plot
	plt.clf()
	plt.close()


# TODO: currently only works for black and white images, must also work for RGB
def visualize_activations(pickle_file, results_path, activations_path):

	# remove old visualization files before creating new ones
	for old_file in listdir(activations_path):
		remove(join(activations_path, old_file))

	# need time of creation to ensure unique filename
	creation_time = time()

	# read content of pickle file
	with open(join(results_path, pickle_file), 'rb') as f:
		content_list = pickle.load(f)

	for layer_no in range(len(content_list)):
		layer_name, layer_activation = content_list[layer_no]

		# print('\nLength of activation shape', layer_activation.shape, 'is', len(layer_activation.shape), '\n')

		# scale to fit between [0.0, 255.0] (image was scaled down to [0.0, 1.0] for network)
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


# for all folders with 'old' counterpart, move to old
def move_to_historical_folder(file_path, filename):

	# find path to main folder
	main_folder = file_path.rsplit('/', 1)[0]

	# calculate correct result number
	result_num = int(len(listdir(join(main_folder, 'old_results'))) / len(listdir(join(main_folder, 'results'))))
	
	# for all folders in same folder as program
	for folder in listdir(main_folder):
		# if there exists an 'old' folder
		if folder.startswith('old_'):
			# move all files in the corresponding 'new' folder to 'old' folder (rename to avoid overwrite)
			for file in listdir(join(main_folder, folder[4:])):
				move(join(main_folder, folder[4:], file), join(main_folder, folder, file.replace('.', '_%s_%d.' % (folder[4:-1], result_num))))
	
	print('\nFiles moved to historical folders\n')


def create_folders(base_path, new_folders):
	for new_folder in new_folders:
		try:
			mkdir(join(base_path, new_folder))
		except FileExistsError:
			pass
