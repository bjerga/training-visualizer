from datetime import date, datetime
from shutil import rmtree
from os import mkdir, listdir
from os.path import join, dirname
from multiprocessing import Process, Value
from urllib.parse import urlencode

from flask import Flask, request, redirect, url_for, render_template, flash, send_from_directory, jsonify
from flask_login import LoginManager, login_required, login_user, logout_user, current_user, abort
import requests
from werkzeug.utils import secure_filename
from sqlalchemy import func, distinct

import subprocess

from .modules.helpers import *
from .modules.models import *
from .modules.forms import *

from visualizer import app

# Create application and get config
#app = Flask(__name__)
#app.config.from_object('visualizer.config')

# initialize database
db.init_app(app)

# initialize login manager
login_manager = LoginManager()
login_manager.init_app(app)

# start bokeh server
bokeh_process = subprocess.Popen(['bokeh', 'serve', '--allow-websocket-origin=localhost:5000',
								  'visualizer/bokeh/training_progress.py',
								  'visualizer/bokeh/layer_activations.py'],
								 stdout=subprocess.PIPE)


# dict to hold {username-key: dict-value{filename-key: list-value[processes]}}
processes = {}


# define method necessary for login manager
@login_manager.user_loader
def load_user(username):
	return User.query.filter_by(username=username).first()


# not sure if this is needed
'''# load default config and override config from an environment variable
app.config.from_envvar('VISUALIZER_SETTINGS', silent=True)'''


# remove old database and create new
@app.cli.command('initdb')
def initdb_command():
	# remove old database and folders
	db.drop_all()
	try:
		rmtree(app.config['UPLOAD_FOLDER'], ignore_errors=True)
	except FileNotFoundError:
		pass
	
	# create new database and folders
	db.create_all()
	try:
		mkdir(app.config['UPLOAD_FOLDER'])
	except FileExistsError:
		pass
	print('Initialized the database')


# define default home page
# shows a list of running files
@app.route('/')
def home():
	if current_user.is_authenticated:
		running = get_running()
		metas = FileMeta.query.filter_by(owner=get_current_user()).filter(FileMeta.filename.in_(running)).all()
		return render_template('home.html', metas=metas)
	return redirect(url_for('login'))


# page to create a new user
@app.route('/create_user', methods=['GET', 'POST'])
def create_user():
	form = CreateUserForm()
	
	# if form values are valid
	if form.validate_on_submit():
		# if chosen username is not unique
		if not unique_username(form.username.data):
			flash('Username is already taken', 'danger')
		else:
			
			# add user to database
			db.session.add(User(form.username.data, form.password.data))
			db.session.commit()
			
			# create folders for user to save programs in
			create_folders(app.config['UPLOAD_FOLDER'], [form.username.data])
			
			# display success message and route to login page
			flash('User successfully created', 'success')
			return redirect(url_for('login'))
		
	return render_template('create_user.html', form=form)


# page to login existing users
@app.route('/login', methods=['GET', 'POST'])
def login():
	form = LoginForm()
	
	# if form values are valid
	if form.validate_on_submit():
		
		# get user from database
		user = User.query.filter_by(username=form.username.data).first()
		
		# if user does not exist or password is invalid
		if not user or not user.check_password(form.password.data):
			flash('Invalid username or password', 'danger')
		else:
			# login valid, authenticate user
			user.authenticated = True
			
			# update user authentication in database
			db.session.add(user)
			db.session.commit()

			# actually log user in
			# set remember=True to enable cookies to remember user
			login_user(user, remember=True)
			flash('You were logged in', 'success')
			
			# TODO: find out how to utilize this
			next_access = request.args.get('next')
			# next_is_valid should check if the user has valid
			# permission to access the `next` url
			if not has_permission(next_access):
				return abort(400)

			# redirect to file list page
			return redirect(next_access or url_for('show_all_files'))
	
	return render_template('login.html', form=form)


# define details for logout
@login_required
@app.route('/logout')
def logout():
	# set current user's authentication to false
	user = current_user
	user.authenticated = False
	
	# update user authentication in database
	db.session.add(user)
	db.session.commit()
	
	# actually log user out
	logout_user()
	flash('You were logged out', 'success')
	
	# redirect to login page
	return redirect(url_for('login'))


# page for uploading file
@login_required
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
	form = FileForm()
	
	# if form values are valid
	if form.validate_on_submit():
		# get file from form
		file = form.file.data
		
		# if actual file and extension is allowed
		if file and allowed_file(file.filename):
			
			# make filename secure
			filename = secure_filename(file.filename)
			
			# TODO: make database model unique and handle database-errors instead of checking uniqueness
			# if filename is unique in database
			if unique_filename(filename):
				
				# create folder for program
				folder_path = join(app.config['UPLOAD_FOLDER'], get_current_user(), filename.rsplit('.', 1)[0])
				try:
					mkdir(folder_path)
				except FileExistsError:
					pass
				
				# create necessary folders within program-folder
				create_folders(folder_path, ['networks', 'results'])
				
				# save program in folder and create file meta
				file_path = join(folder_path, filename)
				file.save(file_path)
				file_meta = FileMeta(filename, date.today().strftime("%d/%m/%y"), file_path, get_current_user())
				
				# create file tags
				for text in form.tags.data:
					file_meta.tags.append(get_existing_tag(text))
				
				# add file name as a tag automatically
				file_meta.tags.append(get_existing_tag(filename.split('.')[0]))
				
				# add file meta to database
				db.session.add(file_meta)
				db.session.commit()
				
				# redirect to file view for uploaded file
				flash('File was successfully uploaded', 'success')
				return redirect(url_for('show_file_overview', filename=filename))

			else:
				flash('Filename already exists', 'danger')
		else:
			flash('File type is not allowed', 'danger')
			
	return render_template('upload_file.html', form=form)


# page for file list view
@login_required
@app.route('/uploads', methods=['GET', 'POST'])
def show_all_files():
	# get all file metas
	metas = FileMeta.query.filter_by(owner=get_current_user()).all()
	
	search_form = SearchForm()
	
	# if form values are valid
	if search_form.validate_on_submit():
		# get search query text
		query = search_form.search.data
		
		# if not empty
		if query:
			# redirect to file list view that only displays matching files
			return redirect(url_for('search', query=query))
		
	return render_template('show_all_files.html', search_form=search_form, metas=metas, running=get_running())


# page for file overview
@login_required
@app.route('/uploads/<filename>/overview', methods=['GET', 'POST'])
def show_file_overview(filename):
	# get information about file
	meta = FileMeta.query.filter_by(filename=filename, owner=get_current_user()).first()

	# get file stored locally
	file_folder = join(app.config['UPLOAD_FOLDER'], get_current_user(), filename.rsplit('.', 1)[0])
	file = send_from_directory(file_folder, filename)
	# check whether the file has produced any results or networks
	has_files = has_associated_files(file_folder)

	# get content of file
	file.direct_passthrough = False
	content = str(file.data, 'utf-8')

	tag_form = TagForm()
	
	# if form values are valid
	if tag_form.validate_on_submit():

		# if there is a tag to be deleted
		text = request.form.get('delete_tag')
		if text:
			# remove tag in database
			tag = Tag.query.filter_by(text=text).first()
			meta.tags.remove(tag)
			db.session.commit()
			return redirect(url_for('show_file_overview', filename=filename))

		# if not, tags should be added
		for text in tag_form.tags.data:
			meta.tags.append(get_existing_tag(text))
		db.session.commit()
		
		# update current page
		return redirect(url_for('show_file_overview', filename=filename))
	
	return render_template('show_file_overview.html', run_form=RunForm(), tag_form=TagForm(),
						   filename=filename, meta=meta, content=content, has_files=has_files)


# page for file visualization view
@login_required
@app.route('/uploads/<filename>/visualization', methods=['GET', 'POST'])
def show_file_visualization(filename):

	# get information about file
	meta = FileMeta.query.filter_by(filename=filename, owner=get_current_user()).first()

	# instantiate the form that is the dropdown menu for selecting visualization
	form = VisualizationForm()

	# get visualization choice from dropdown, or default to the first of the list
	if form.validate_on_submit():
		visualization_path = request.form.get('visualization')
	else:
		visualization_path = form.visualization.choices[0][0]

	#TODO: save the url for the server in a config
	# build the url for getting a certain visualization technique given a user and file
	params = {'user': get_current_user(), 'file': filename.split('.')[0]}
	url = app.config['BOKEH_SERVER'] + visualization_path + '?' + urlencode(params)
	# send a GET request to the bokeh server
	plot = requests.get(url).content.decode('ascii')

	return render_template('show_file_visualization.html', filename=filename, meta=meta, plot=plot, form=form)


# page for training progress view
@login_required
@app.route('/uploads/<filename>/training_progress')
def show_file_training_progress(filename):

	# get information about file
	meta = FileMeta.query.filter_by(filename=filename, owner=get_current_user()).first()

	#TODO: save the url for the server in a config
	# build the url for getting a certain visualization technique given a user and file
	params = {'user': get_current_user(), 'file': filename.split('.')[0]}
	url = 'http://localhost:5006/training_progress?' + urlencode(params)
	# send a GET request to the bokeh server
	plot = requests.get(url).content.decode('ascii')

	return render_template('show_file_training_progress.html', filename=filename, meta=meta, plot=plot)


# define how to run a program using new processes
@login_required
@app.route('/uploads/<filename>/run', methods=['POST'])
def run_upload(filename):
	global processes

	# get information about file
	meta = FileMeta.query.filter_by(filename=filename, owner=get_current_user()).first()
	print('\n\nNew thread started for %s\n\n' % meta.path)

	result_path = meta.path.replace(filename, 'results')

	# remove results folder and all its files before creating a new, empty one
	try:
		rmtree(result_path, ignore_errors=True)
	except FileNotFoundError:
		pass
	finally:
		mkdir(result_path)

	# clear process-list for filename
	prevent_process_key_error(filename)
	processes[get_current_user()][filename] = []

	# shared boolean denoting if run_python_shell-process is writing
	shared_bool = Value('i', True)
	
	# start and save a new process for running the program
	p = Process(target=run_python_shell, args=(meta.path, shared_bool))
	p.start()

	processes[get_current_user()][filename].append(p)

	# update last run column in database
	meta.last_run_date = datetime.now().strftime("%d/%m/%y %H:%M")
	db.session.commit()
	
	# redirect to file training progress view
	return redirect(url_for('show_file_training_progress', filename=filename))


# define how to download a trained network
@login_required
@app.route('/uploads/<filename>/download')
def download_network(filename):
	network_folder = join(app.config['UPLOAD_FOLDER'], get_current_user(), filename.rsplit('.', 1)[0], 'networks')
	network_name = listdir(network_folder)[-1]
	return send_from_directory(network_folder, network_name, as_attachment=True)


# define how to delete file
@login_required
@app.route('/uploads/<filename>/delete')
def delete_file(filename):

	# delete file information from database
	meta = FileMeta.query.filter_by(filename=filename, owner=get_current_user()).first()
	db.session.delete(meta)
	db.session.commit()

	# delete the folder of the file to be deleted
	rmtree(join(app.config['UPLOAD_FOLDER'], get_current_user(), filename.rsplit('.', 1)[0]), ignore_errors=True)

	# redirect to file list view
	flash(filename + ' was deleted', 'danger')
	return redirect(url_for('show_all_files'))


# define how to find for search results
@login_required
@app.route('/search_results/<query>')
def search(query):
	# find all files that match search results
	tags = query.split(" ")
	results = FileMeta.query.join(FileMeta.tags).filter(FileMeta.owner == get_current_user()).filter(Tag.text.in_(tags))\
		.group_by(FileMeta).having(func.count(distinct(Tag.id)) == len(tags))
	
	# redirect to file list view, passing only files that match search
	return render_template('show_all_files.html', search_form=SearchForm(), metas=results, search_text=query)


# helper method dependent on app
# check if there exists at least one network for selected file
@login_required
@app.route('/uploads/<filename>/check_networks_exist')
def check_networks_exist(filename):
	networks_exist = False

	network_folder = join(app.config['UPLOAD_FOLDER'], get_current_user(), filename.rsplit('.', 1)[0], 'networks')
	if listdir(network_folder):
		networks_exist = True

	return jsonify(networks_exist=networks_exist)


# define how to check if file is running
@login_required
@app.route('/uploads/<filename>/check_running')
# check if file is running, return json-object
def check_running(filename):
	return jsonify(is_running=is_running(filename))


# helper method dependent on app
# check if file is running
def is_running(filename):
	prevent_process_key_error(filename)

	is_file_running = False

	# if any process for the file is still alive, return true
	for process in processes[get_current_user()][filename]:
		if process.is_alive():
			is_file_running = True
			break

	return is_file_running


@login_required
@app.route('/uploads/running')
def get_running_json():
	return jsonify(running=list(get_running()))


# return a set of all files running
def get_running():
	running = set()
	try:
		user_processes = processes[get_current_user()]
	except KeyError:
		user_processes = {}

	for filename in user_processes:
		for process in user_processes[filename]:
			if process.is_alive():
				running.add(filename)

	return running


# helper method dependent on app
# used to prevent key errors for process dict
def prevent_process_key_error(filename):
	global processes
	try:
		# test if username is in process dict
		processes[get_current_user()]
	except KeyError:
		# if not, add it
		processes[get_current_user()] = {}

	try:
		# test if filename in user-process dict
		processes[get_current_user()][filename]
	except KeyError:
		# if not, add it
		processes[get_current_user()][filename] = []
