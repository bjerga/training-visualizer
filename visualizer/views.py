from visualizer.forms import *
from visualizer.utils import *
from visualizer.models import *

from datetime import date, datetime
from shutil import rmtree
from os import mkdir, listdir, remove
from os.path import join, getmtime, split, basename, dirname

from flask import request, redirect, url_for, render_template, flash, send_from_directory, jsonify, abort
from flask_login import login_required, login_user, logout_user, current_user
from werkzeug.utils import secure_filename
from sqlalchemy import func, distinct

from tailer import tail

# Import the database, application, and login_manager object from the main visualizer module
from visualizer import db, app, login_manager

# dict to hold {username-key: dict-value{filename-key: process}}
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
		metas = []
		if running:
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
			
			# redirect to file list page
			return redirect(url_for('show_all_files'))
	
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

			# if filename is unique in database
			if unique_filename(filename):
				
				# create folder for program
				folder_path = get_file_folder(filename)
				try:
					mkdir(folder_path)
				except FileExistsError:
					pass
				
				# create necessary folders within program-folder
				create_folders(folder_path, ['networks', 'results', 'images'])
				
				# save program in folder and create file meta
				file_path = join(folder_path, filename)
				file.save(file_path)
				file_meta = FileMeta(filename, date.today().strftime("%d/%m/%y"), file_path, get_current_user())
				
				# create file tags
				for text in form.tags.data:
					file_meta.tags.append(get_existing_tag(text))
				
				# add file name as a tag automatically
				file_meta.tags.append(get_existing_tag(get_wo_ext(filename)))
				
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
@app.route('/uploads/<filename>', methods=['GET', 'POST'])
def show_file_overview(filename):
	# get information about file
	meta = FileMeta.query.filter_by(filename=filename, owner=get_current_user()).first()

	running = is_running(filename)

	# get file stored locally
	file = send_from_directory(get_file_folder(filename), filename)
	# check whether the file has produced any results or networks
	has_files = has_associated_files(filename)
	has_image = has_visualization_image(filename)


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
	
	return render_template('show_file_overview.html', run_form=RunForm(), tag_form=TagForm(), is_running=running,
						   filename=filename, meta=meta, content=content, has_files=has_files, has_image=has_image,
						   visualizations=app.config['VISUALIZATIONS'])


@app.route('/uploads/<filename>/visualization_image')
def send_visualization_image(filename):
	path = get_visualization_img_abs_path(filename)
	return send_from_directory(dirname(path), basename(path))


# page for file visualization view
@login_required
@app.route('/uploads/<filename>/<visualization>', methods=['GET', 'POST'])
def show_file_visualization(filename, visualization):

	# get information about file
	meta = FileMeta.query.filter_by(filename=filename, owner=get_current_user()).first()

	# get the correct plot given the filename and type of visualization
	plot = get_bokeh_plot(filename, visualization)

	return render_template('show_file_visualization.html', filename=filename, meta=meta, plot=plot, visualizations=app.config['VISUALIZATIONS'])


# define how to run a program using new processes
@login_required
@app.route('/uploads/<filename>/run', methods=['POST'])
def run_upload(filename):
	global processes

	form = RunForm()

	if form.validate_on_submit():

		# get information about file
		meta = FileMeta.query.filter_by(filename=filename, owner=get_current_user()).first()

		# get image from form
		image = form.image.data

		image_path = get_visualization_img_abs_path(filename)

		# flask error if no image is selected and there is no previously uploaded image
		if image.filename is '' and image_path is None:
			flash('You need to select an image', 'danger')
			return redirect(url_for('show_file_overview', filename=filename))

		# if a new image has been uploaded
		if image.filename is not '':

			# make sure the old image is deleted (will cause problems if it has a different format than the new one)
			if image_path is not None:
				remove(image_path)

			# flash error if image format is not allowed
			if not allowed_image(image.filename):
				flash('File type is not allowed', 'danger')
				return redirect(url_for('show_file_overview', filename=filename))

			# save image as 'image' with the correct ending and make name secure
			image_name = secure_filename(image.filename)
			image.save(join(get_images_folder(filename), image_name))

		result_folder = get_results_folder(filename)

		# remove results folder and all its files before creating a new, empty one
		try:
			rmtree(result_folder, ignore_errors=True)
		except FileNotFoundError:
			pass
		finally:
			mkdir(result_folder)

		# remove process for filename
		remove_process(filename)

		print('\n\nNew thread started for %s\n\n' % meta.path)

		# start and save a new subprocess for running the program
		p = run_python_shell(meta.path)
		add_process(filename, p)

		# update last run column in database
		meta.last_run_date = datetime.now().strftime("%d/%m/%y %H:%M")
		db.session.commit()

	return redirect(url_for('show_file_overview', filename=filename))


@login_required
@app.route('/uploads/<filename>/stop')
def stop_file(filename):
	# get process and kill and remove it if it is running
	p = get_process(filename)
	if p is not None:
		p.kill()
		remove_process(filename)
		flash(filename + ' was stopped', 'danger')
	return redirect(url_for('show_file_overview', filename=filename))


# page for file output view
@login_required
@app.route('/uploads/<filename>/output', methods=['GET', 'POST'])
def show_file_output(filename):
	# get information about file
	meta = FileMeta.query.filter_by(filename=filename, owner=get_current_user()).first()
	running = is_running(filename)
	return render_template('show_file_output.html', filename=filename, meta=meta, is_running=running,
						   visualizations=app.config['VISUALIZATIONS'])


# returns the last x lines of CLI output for a specific user's specific file, based on config value
@login_required
@app.route('/cli_output/<user>/<filename>')
def get_cli_output(user, filename):
	try:
		output = '\n'.join(tail(open(get_output_file(user, filename)), app.config['NO_OF_OUTPUT_LINES']))
	except FileNotFoundError:
		# if output file has not yet been created, return string saying this
		output = 'No output has been produced.'
	return jsonify(output=output)


# define how to download the whole CLI output file
@login_required
@app.route('/cli_output/<user>/<filename>/download')
def download_cli_output(user, filename):
	output_folder, output_name = split(get_output_file(user, filename))
	# give the output file a more descriptive name on the form 'filename_output.txt'
	attachment_filename = '_'.join((get_wo_ext(filename), output_name))
	return send_from_directory(output_folder, output_name, as_attachment=True, attachment_filename=attachment_filename)


# define how to download a trained network
@login_required
@app.route('/uploads/<filename>/download')
def download_network(filename):
	network_folder = get_networks_folder(filename)
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
	rmtree(get_file_folder(filename), ignore_errors=True)

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
	networks_exist = listdir(get_networks_folder(filename))
	return jsonify(networks_exist=bool(networks_exist))


# define how to check if file is running
@login_required
@app.route('/uploads/<filename>/check_running')
# check if file is running, return json-object
def check_running(filename):
	return jsonify(is_running=is_running(filename))


# helper method dependent on app
# check if file is running
def is_running(filename):
	p = get_process(filename)
	# if the process for the file is still alive, return true
	if p is not None:
		return p.poll() is None
	return False


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

	# check which files are running and add them to the set of running files
	for filename in user_processes:
		p = get_process(filename)
		if p is not None and p.poll() is None:
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
		processes[get_current_user()][filename] = None


def add_process(filename, p):
	prevent_process_key_error(filename)
	processes[get_current_user()][filename] = p


def remove_process(filename):
	prevent_process_key_error(filename)
	processes[get_current_user()][filename] = None


def get_process(filename):
	prevent_process_key_error(filename)
	return processes[get_current_user()][filename]
