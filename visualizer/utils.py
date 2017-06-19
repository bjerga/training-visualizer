import subprocess
from sys import exc_info
from os import listdir, mkdir
from os.path import join, relpath, basename
from urllib.parse import urlencode, urljoin

from flask import url_for
from flask_login import current_user

from bokeh.embed import autoload_server

from visualizer.models import User, Tag, FileMeta
from visualizer.config import UPLOAD_FOLDER, PYTHON, BOKEH_SERVER


# allowed extensions for uploading files
ALLOWED_FILE_EXTENSIONS = {'py'}
# allowed extensions for uploading images to use for visualization
ALLOWED_IMAGE_EXTENSIONS = {'jpeg', 'jpg', 'png'}


# get current user in form of string
def get_current_user():
	return str(current_user)


# check is username is unique in database
def unique_username(username):
	if User.query.filter_by(username=username).first():
		return False
	return True


# check if filename is unique for current user in database
def unique_filename(filename):
	if FileMeta.query.filter_by(filename=filename, owner=get_current_user()).first():
		return False
	return True


def get_wo_ext(filename):
	return filename.rsplit('.', 1)[0]


def get_ext(filename):
	return filename.rsplit('.', 1)[1]


# check if file extension is allowed
def allowed_file(filename):
	return '.' in filename and get_ext(filename).lower() in ALLOWED_FILE_EXTENSIONS


# check if file extension is allowed for image
def allowed_image(filename):
	return '.' in filename and get_ext(filename).lower() in ALLOWED_IMAGE_EXTENSIONS


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


# check if a network has networks or visualization data associated with it
def has_associated_files(filename):
	return listdir(get_networks_folder(filename)) or listdir(get_results_folder(filename))


# check if a network has previously uploaded input image associated with it
def has_visualization_image(filename):
	if get_visualization_img_abs_path(filename):
		return True
	return False


# get path of the folder of a certain file
def get_file_folder(filename):
	return join(UPLOAD_FOLDER, get_current_user(), get_wo_ext(filename))


# get path of the results folder of a certain file
def get_results_folder(filename):
	return join(get_file_folder(filename), 'results')


# get path of the networks folder of a certain file
def get_networks_folder(filename):
	return join(get_file_folder(filename), 'networks')


# get path of the images folder of a certain file
def get_images_folder(filename):
	return join(get_file_folder(filename), 'images')


# return absolute path of visualization image, or none if no image has been uploaded
def get_visualization_img_abs_path(filename):
	images_folder = get_images_folder(filename)
	images = listdir(images_folder)
	if images:
		return join(images_folder, images[-1])
	return None


def get_output_file(user, filename):
	return join(UPLOAD_FOLDER, user, get_wo_ext(filename), 'output.txt')


# run a python program via command line
def run_python_shell(file_path):
	if file_path:
		
		print('\nSubprocess started\n')

		# run program via command line
		with open(get_output_file(get_current_user(), basename(file_path)), 'w') as f:
			try:
				p = subprocess.Popen([PYTHON, file_path], stdout=f)
			except FileNotFoundError as e:
				file_folder, file_name = file_path.rsplit('\\', 1)
				raise FileNotFoundError(str(e) + '. Please make sure that file {!r} exists in folder {!r}. Otherwise, '
												 'problem might be caused by system not recognizing {!r} as a command. '
												 'If so, please alter PYTHON variable in config.py to match your Python '
												 'command word.'
												 ''.format(file_name, file_folder, PYTHON)).with_traceback(exc_info()[2])
		return p
	else:
		print('\n\nNo file found\n\n')
	
	# make sure process ends (not certain this is needed)
	return


# create all folders in list at the specified base path
def create_folders(base_path, new_folders):
	for new_folder in new_folders:
		try:
			mkdir(join(base_path, new_folder))
		except FileExistsError:
			pass


def get_bokeh_plot(filename, app_path):
	# get the script for a given visualization from the bokeh server
	script = autoload_server(model=None, url=urljoin(BOKEH_SERVER, app_path))

	# set the correct query parameters
	params = {'user': get_current_user(), 'file': get_wo_ext(filename)}
	
	# manually add the query parameters to the script, this is not yet implemented in bokeh
	script_list = script.split('\n')
	script_list[2] = script_list[2][:-1]
	script_list[2] += '&' + urlencode(params) + '"'
	
	return '\n'.join(script_list)
