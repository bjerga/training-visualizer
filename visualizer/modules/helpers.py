import re, os
import subprocess as sub
import matplotlib.pyplot as plt
from time import time, sleep
from shutil import move

from flask import abort
from flask_login import current_user

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


def plot_callback_output(file_path, filename, shared_bool):

	plot_colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w']

	# create file path for plots
	results_path = file_path.replace(filename, 'results')
	plots_path = file_path.replace(filename, 'plots')
	print('\nPlot folder path: %s\n' % plots_path)

	output_files = []
	while not output_files:
		print('\nResults file not found, waiting for creation...\n')
		sleep(3)
		output_files = os.listdir(results_path)

	output_file_paths = {output_file.rsplit('.', 1)[0]: os.path.join(results_path, output_file) for output_file in output_files}

	outputs = {}

	# if other process is writing
	while shared_bool.value:

		# get length of old error values
		old_length = sum([len(values) for values in outputs.values()])

		# read outputs
		for output_name in output_file_paths.keys():
			with open(output_file_paths[output_name], 'r') as f:
				outputs[output_name] = [float(line) for line in f]

		# if new output values read, plot and save accuracy error so far
		new_length = sum([len(values) for values in outputs.values()])
		if new_length != old_length:

			# remove old plots
			for file in os.listdir(plots_path):
				os.remove(os.path.join(plots_path, file))

			# create new plots
			plot_num = 0
			for output_name in outputs.keys():

				x_label, y_label, model_no = output_name.split('_')

				# plt.figure(figsize=(20, 10))
				plt.plot(outputs[output_name], plot_colors[plot_num] + '-')
				plt.title(('%s Over %s For Model No. %s' % (y_label, x_label, model_no)).title())
				plt.xlabel(x_label.title())
				plt.ylabel(y_label.title())

				# set limits of x to be outermost points
				plt.xlim([0, len(outputs[output_name])-1])

				# save new plots
				plt.savefig(os.path.join(plots_path, '%s_plot_%d.png' % (output_name, time())))

				# clear figure for next plot
				plt.clf()

				plot_num += 1

			# close plots
			plt.close()

		# sleep to keep from constantly plotting
		sleep(3)

	print('\nPlotting done\n')

	# make sure process ends (not certain this is needed)
	return
	
	
def move_to_historical_folder(file_path, filename):
	
	results_path = file_path.replace(filename, 'results')
	plots_path = file_path.replace(filename, 'plots')
	
	# move results and plots to old_results and old_plots, respectively
	old_results_path = file_path.replace(filename, 'old_results')
	result_num = len(os.listdir(old_results_path))
	# result_num = int(np.sum([1 for file in os.listdir(base_write_path) if file.startswith('mnist_nn_' + error_type)]))
	for file in os.listdir(results_path):
		move(os.path.join(results_path, file), os.path.join(old_results_path, file.replace('.', '_result_%d.' % result_num)))
		
	old_plots_path = file_path.replace(filename, 'old_plots')
	for file in os.listdir(plots_path):
		move(os.path.join(plots_path, file), os.path.join(old_plots_path, file.replace('_plot', '_result_%d_plot' % result_num)))
	
	print('\nFiles moved to historical folders\n')


def create_folders(base_path, new_folders):
	for new_folder in new_folders:
		try:
			os.mkdir(os.path.join(base_path, new_folder))
		except FileExistsError:
			pass
