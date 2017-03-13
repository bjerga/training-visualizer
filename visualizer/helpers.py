import subprocess as sub
from os import listdir, mkdir
from os.path import join

from flask_login import current_user

from visualizer.models import User, Tag, FileMeta


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


# create all folders in list at the specified base path
def create_folders(base_path, new_folders):
	for new_folder in new_folders:
		try:
			mkdir(join(base_path, new_folder))
		except FileExistsError:
			pass
