from datetime import date
from shutil import rmtree
from os import mkdir, listdir
from os.path import join, dirname
from multiprocessing import Process, Value

from flask import Flask, request, redirect, url_for, render_template, flash, send_from_directory, jsonify
from flask_login import LoginManager, login_required, login_user, logout_user, current_user, abort
from werkzeug.utils import secure_filename
from sqlalchemy import func, distinct

from .modules.helpers import *
from .modules.models import *
from .modules.forms import *

# Create application
app = Flask(__name__)
app.config.from_object(__name__)

# temporary secret key
app.secret_key = 'thisissupersecretestkeyintheworld'

# entry to hold path to upload folder
app.config['UPLOAD_FOLDER'] = join(dirname(__file__), 'static', 'user_storage')

# dict to hold {username-key: dict-value{filename-key: list-value[processes]}}
app.config['processes'] = {}

# entry to hold database connection
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///visualizer.db'

# set to disable notifications of overhead when running
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# initialize database
db.init_app(app)

# initialize login manager
login_manager = LoginManager()
login_manager.init_app(app)


# define method necessary for login manager
@login_manager.user_loader
def load_user(username):
	return User.query.filter_by(username=username).first()

# load default config and override config from an environment variable
app.config.from_envvar('VISUALIZER_SETTINGS', silent=True)


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
@app.route('/')
def home():
	if current_user.is_authenticated:
		return redirect(url_for('show_all_files'))
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
			
			# create folders for user to save data in
			create_folders(app.config['UPLOAD_FOLDER'], [form.username.data, form.username.data + '/programs',
														 form.username.data + '/data'])
			
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
				folder_path = join(app.config['UPLOAD_FOLDER'], get_current_user(), 'programs', filename.rsplit('.', 1)[0])
				try:
					mkdir(folder_path)
				except FileExistsError:
					pass
				
				# create necessary folders within program-folder
				create_folders(folder_path, ['results', 'networks', 'old_results', 'plots', 'old_plots', 'activations', 'old_activations'])
				
				# save program in folder and create file meta
				file_path = join(folder_path, filename)
				file.save(file_path)
				file_meta = FileMeta(filename, date.today(), file_path, get_current_user())
				
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
				return redirect(url_for('show_file_code', filename=filename))

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
		
	return render_template('show_all_files.html', search_form=search_form, metas=metas)


# page for file code view
@login_required
@app.route('/uploads/<filename>/code', methods=['GET', 'POST'])
def show_file_code(filename):
	# get information about file
	meta = FileMeta.query.filter_by(filename=filename, owner=get_current_user()).first()
	
	# get file stored locally
	file = send_from_directory(join(app.config['UPLOAD_FOLDER'], get_current_user(), 'programs', filename.rsplit('.', 1)[0]), filename)
	
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
			return redirect(url_for('show_file_code', filename=filename))

		# if not, tags should be added
		for text in tag_form.tags.data:
			meta.tags.append(get_existing_tag(text))
		db.session.commit()
		
		# update current page
		return redirect(url_for('show_file_code', filename=filename))
	
	return render_template('show_file_code.html', form=RunForm(), tag_form=TagForm(), filename=filename, meta=meta, content=content)


# page for file visualization view
@login_required
@app.route('/uploads/<filename>/visualization', methods=['GET', 'POST'])
def show_file_visualization(filename):
	# get information about file and visualize
	meta = FileMeta.query.filter_by(filename=filename, owner=get_current_user()).first()
	return render_template('show_file_visualization.html', filename=filename, meta=meta)


# page for file history view
@login_required
@app.route('/uploads/<filename>/history')
def show_file_history(filename):
	# get information about file and show history
	meta = FileMeta.query.filter_by(filename=filename, owner=get_current_user()).first()
	return render_template('show_file_history.html', filename=filename, meta=meta)


# define how to get visualization sources
# returns json-object
@login_required
@app.route('/uploads/<filename>/visualization_sources')
def get_visualization_sources(filename,):
	
	# get folder name from filename
	folder_name = filename.rsplit('.', 1)[0]
	
	# for plots
	plot_sources = []
	
	# get all static plot URLs
	plots = sorted(listdir(join(app.config['UPLOAD_FOLDER'], get_current_user(), 'programs', folder_name, 'plots')))
	for plot in plots:
		plot_path = 'user_storage/%s/programs/%s/plots/%s' % (get_current_user(), folder_name, plot)
		plot_sources.append(url_for('static', filename=plot_path))
	
	# for activations
	# each tuple contain sources for all activation images for one layer
	activation_tuples = []
	
	# get all static activation URLs and add to correct tuple
	activations = sorted(listdir(join(app.config['UPLOAD_FOLDER'], get_current_user(), 'programs', folder_name, 'activations')))
	prev_activation_layer = ''
	for activation in activations:
		
		activation_path = 'user_storage/%s/programs/%s/activations/%s' % (get_current_user(), folder_name, activation)
		
		# every activation starts with 'ln' and then a number, denoting layer number
		# if activation belongs to new layer, add new (layer title, act. path list)-tuple
		if activation[2] != prev_activation_layer:
			layer_title = 'Layer %s - %s' % (activation[2], activation.split('_', 2)[1].title())
			activation_tuples.append((layer_title, [url_for('static', filename=activation_path)]))
			prev_activation_layer = activation[2]
		# if not, append to act. path list for latest layer number
		else:
			activation_tuples[-1][1].append(url_for('static', filename=activation_path))
	
	# use is_running to investigate if visualization producing process is still running
	return jsonify(plot_sources=plot_sources, activation_tuples=activation_tuples, should_visualize=is_running(filename))


# define how to run a program using new processes
@login_required
@app.route('/uploads/<filename>/run', methods=['POST'])
def run_upload(filename):
	# get information about file
	meta = FileMeta.query.filter_by(filename=filename, owner=get_current_user()).first()
	print('\n\nNew thread started for %s\n\n' % meta.path)
	
	# move results and plots if any exist
	if len(listdir(meta.path.replace(meta.filename, 'results'))) != 0:
		move_to_historical_folder(meta.path)

	# clear process-list for filename
	prevent_process_key_error(filename)
	app.config['processes'][get_current_user()][filename] = []
	
	# shared boolean denoting if run_python_shell-process is writing
	shared_bool = Value('i', True)
	
	# start and save a new process for running the program
	p = Process(target=run_python_shell, args=(meta.path, shared_bool))
	p.start()
	app.config['processes'][get_current_user()][filename].append(p)

	# start and save a new process for visualizing the callback output
	p = Process(target=visualize_callback_output, args=(meta.path, meta.filename, shared_bool))
	p.start()
	app.config['processes'][get_current_user()][filename].append(p)
	
	# redirect to file visualization view
	return redirect(url_for('show_file_visualization', filename=filename))


# define how to download a trained network
@login_required
@app.route('/uploads/<filename>/download')
def download_network(filename):

	network_folder = join(app.config['UPLOAD_FOLDER'], get_current_user(), 'programs', filename.rsplit('.', 1)[0], 'networks')

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
	rmtree(join(app.config['UPLOAD_FOLDER'], get_current_user(), 'programs', filename.rsplit('.', 1)[0]),
		   ignore_errors=True)

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

	network_folder = join(app.config['UPLOAD_FOLDER'], get_current_user(), 'programs', filename.rsplit('.', 1)[0], 'networks')
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
# if running, return 1, else -1
def is_running(filename):
	prevent_process_key_error(filename)

	is_file_running = -1

	# if any process for the file is still alive, return 1
	for process in app.config['processes'][get_current_user()][filename]:
		if process.is_alive():
			is_file_running = 1
			break

	return is_file_running


# helper method dependent on app
# used to prevent key errors for process dict
def prevent_process_key_error(filename):
	try:
		# test if username is in process dict
		app.config['processes'][get_current_user()]
	except KeyError:
		# if not, add it
		app.config['processes'][get_current_user()] = {}

	try:
		# test if filename in user-process dict
		app.config['processes'][get_current_user()][filename]
	except KeyError:
		# if not, add it
		app.config['processes'][get_current_user()][filename] = []
