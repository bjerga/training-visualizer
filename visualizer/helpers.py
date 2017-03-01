import subprocess
from os import listdir, mkdir
from os.path import join, relpath, basename, dirname

import sys

from flask import url_for
from flask_login import current_user

from visualizer.models import User, Tag, FileMeta
from visualizer.config import UPLOAD_FOLDER


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


# to be implemented
def has_permission(next_access):
	# TODO: implement
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
def has_associated_files(filename):
	return listdir(get_networks_folder(filename)) or listdir(get_results_folder(filename))


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


# return relative path of visualization image, or none if no image has been uploaded
def get_visualization_img_rel_path(filename):
	abs_image_path = get_visualization_img_abs_path(filename)
	if abs_image_path:
		rel_image_path = relpath(abs_image_path, UPLOAD_FOLDER)
		# needs to add 'static' and 'user_storage' to the path
		return url_for('static', filename=join(basename(UPLOAD_FOLDER), rel_image_path))
	return None


# return absolute path of visualization image, or none if no image has been uploaded
def get_visualization_img_abs_path(filename):
	images_folder = get_images_folder(filename)
	images = listdir(images_folder)
	if images:
		return join(images_folder, images[-1])
	return None


# run a python program via command line
def run_python_shell(file_path):
	if file_path:
		
		print('\nSubprocess started\n')

		# get PYTHONPATH to send to subprocess so that it has visualizer registered as a module
		python_path = ":".join(sys.path)[1:]
		
		# run program via command line
		with open(join(dirname(file_path), 'output.txt'), 'w') as f:
			subprocess.Popen('python3 ' + file_path, shell=True, env={'PYTHONPATH': python_path}, stdout=f).wait()

		print('\nSubprocess finished\n')
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
