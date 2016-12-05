import re, os
import subprocess as sub
import matplotlib.pyplot as plt
from time import time, sleep
from shutil import move

from flask import abort
from flask_login import current_user

from visualizer.modules.models import User, Tag, FileMeta


ALLOWED_EXTENSIONS = {'py'}


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


def plot_accuracy_error(file_path, filename, shared_bool):
	
	# create file path for plots
	results_path = file_path.replace(filename, 'results')
	plots_path = file_path.replace(filename, 'plots')
	print('\nPlot folder path: %s\n' % plots_path)
	
	error_filename = ''
	while True:
		sleep(10)
		try:
			error_filename = os.listdir(results_path)[-1]
			break
		# if file has not been made yet, sleep and try again
		except IndexError:
			print('\nPlot file not found, waiting for creation...\n')
			
	error_file_path = os.path.join(results_path, error_filename)
	
	errors = []
	plot_num = 0
	# if other process is writing
	while shared_bool.value:
		sleep(10)
		
		# copy old errors
		old_errors = errors[:]
		
		errors = read_error_file(error_file_path)
		
		print('Errors so far:', errors)
		# if new errors read, plot and save accuracy error so far
		if len(errors) != len(old_errors):
			# plt.figure(figsize=(20, 10))
			plt.plot(errors, 'ro-', label='Error')
			plt.legend(loc='upper right')
			plt.title('Accuracy Error Over Epochs')
			plt.xlabel('Epoch')
			plt.ylabel('Accuracy Error')
			
			# hard-coded
			plt.xlim([0, 10])
			plt.ylim([0, 10])
			
			plt.savefig(os.path.join(plots_path, '%s_plot_%d_%d.png' % (error_filename.rsplit('.', 1)[0], plot_num, time())))
			# plt.show()
			plt.clf()
			plt.close()
			
			print('\nSaved accuracy error plot as no. %d\n' % plot_num)
			plot_num += 1
	
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


def read_error_file(error_file_path):
	
	errors = []
	with open(error_file_path, 'r') as f:
		for line in f:
			if line != '':
				errors.append(float(line))
	
	return errors


def create_folders(base_path, new_folders):
	for new_folder in new_folders:
		try:
			os.mkdir(os.path.join(base_path, new_folder))
		except FileExistsError:
			pass
