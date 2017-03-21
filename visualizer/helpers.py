import subprocess as sub
from os import listdir, mkdir
from os.path import join

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


# check if file extension is allowed
def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_FILE_EXTENSIONS


# check if file extension is allowed for image
def allowed_image(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS


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


# return relative path of visualization image, or none if no image has been uploaded
def get_visualization_img_rel_path(filename):
	file_folder = join(UPLOAD_FOLDER, get_current_user(), get_wo_ext(filename))
	for name in listdir(file_folder):
		if 'image' in name:
			return url_for('static', filename=join('user_storage', get_current_user(), get_wo_ext(filename), name))
	return None


# return absolute path of visualization image, or none if no image has been uploaded
def get_visualization_img_abs_path(filename):
	file_folder = join(UPLOAD_FOLDER, get_current_user(), get_wo_ext(filename))
	for name in listdir(file_folder):
		if 'image' in name:
			return join(file_folder, name)
	return None


# run a python program via command line
def run_python_shell(file_path):
	if file_path:
		
		print('\nSubprocess started\n')

		# get PYTHONPATH to send to subprocess so that it has visualizer registered as a module
		python_path = ":".join(sys.path)[1:]
		
		# run program via command line
		sub.run('python3 ' + file_path, shell=True, env={'PYTHONPATH': python_path})

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
