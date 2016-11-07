import re
import subprocess as sub
import matplotlib.pyplot as plt
from time import sleep

from flask import abort
from flask_login import current_user

from modules.models import User


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


def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


# TODO: authorization could be improved
def check_authorization(username):
	if username != str(current_user):
		abort(401)
		

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
		sub.run('python ' + file_path)
		
		print('\nSubprocess finished\n')
	else:
		print('\n\nNo file found\n\n')
	
	# mark via shared boolean that process is no longer writing
	shared_bool.value = False
	
	# make sure process ends (not certain this is needed)
	return


def plot_accuracy_error(file_path, shared_bool):

	# create file path for plots
	plot_file_path = file_path.replace('programs', 'plots').replace('.py', '_result.txt')
	print('\nPlot file path: %s\n' % plot_file_path)
	
	accuracy_errors = []
	num = 0
	# if other process is writing
	while shared_bool.value:
		sleep(6)
		
		# copy old errors
		old_accuracy_errors = accuracy_errors[:]
		
		try:
			accuracy_errors = read_plot_file(plot_file_path)
		# if file has not been made yet, sleep and try again
		except FileNotFoundError:
			print('\nFile not found\n')
			continue
			
		print('Errors so far:', accuracy_errors)
		# if new errors read, plot and save accuracy error so far
		if len(accuracy_errors) != len(old_accuracy_errors):
			# plt.figure(figsize=(20, 10))
			plt.plot(accuracy_errors, 'r-', label='Error')
			plt.legend(loc='upper right')
			plt.title('Accuracy Error Over Epochs')
			plt.xlabel('Epoch')
			plt.ylabel('Accuracy Error')
			
			# hard-coded
			plt.xlim([0, 10])
			plt.ylim([0, 10])
			
			plt.savefig(plot_file_path.replace('.txt', '_%d.png' % num))
			# plt.show()
			plt.clf()
			plt.close()
			
			print('\nSaved accuracy error plot as no. %d\n' % num)
			num += 1
			
	print('\nPlotting done\n')
		
	# make sure process ends (not certain this is needed)
	return


def read_plot_file(plot_file_path):
	
	accuracy_errors = []
	with open(plot_file_path, 'r') as f:
		for line in f:
			if line != '':
				accuracy_errors.append(float(line))
	
	return accuracy_errors
