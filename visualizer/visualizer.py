import os
from datetime import date
from shutil import rmtree
from multiprocessing import Process, Value

import flask_login as fl
from flask import Flask, request, redirect, url_for, render_template, flash, send_from_directory, jsonify
from flask_login import login_required
from werkzeug.utils import secure_filename
from sqlalchemy import func, distinct

from .modules.helpers import *
from .modules.models import *
from .modules.forms import *

# Create application
app = Flask(__name__)
app.config.from_object(__name__)
app.secret_key = 'thisissupersecretestkeyintheworld'

app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'static', 'user_storage')
# app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'user_storage')

app.config['processes'] = {}

# app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///visualizer.db'
# set to disable notifications of overhead when running
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

login_manager = fl.LoginManager()
login_manager.init_app(app)


@login_manager.user_loader
def load_user(username):
	return User.query.filter_by(username=username).first()

# Load default config and override config from an environment variable
# app.config.update(dict(
# 	SECRET_KEY='thisissupersecret',
# ))
app.config.from_envvar('VISUALIZER_SETTINGS', silent=True)


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
		os.mkdir(app.config['UPLOAD_FOLDER'])
	except FileExistsError:
		pass
	print('Initialized the database')


@app.route('/create_user', methods=['GET', 'POST'])
def create_user():
	form = CreateUserForm()
	if form.validate_on_submit():
		if not unique_username(form.username.data):
			flash('Username is already taken', 'danger')
		else:
			# add user to database
			db.session.add(User(form.username.data, form.password.data))
			db.session.commit()
			
			# create folders for user to save data in
			create_folders(app.config['UPLOAD_FOLDER'], [form.username.data, form.username.data + '/programs',
														 form.username.data + '/data'])
			
			flash('User successfully created', 'success')
			return redirect(url_for('login'))
	return render_template('create_user.html', form=form)


@app.route('/', methods=['GET', 'POST'])
def login():
	form = LoginForm()
	if form.validate_on_submit():
		user = User.query.filter_by(username=form.username.data).first()
		# user = load_user(form.username.data)
		if not user:
			flash('Invalid username', 'danger')
		elif not user.check_password(form.password.data):
			flash('Invalid password', 'danger')
		else:
			user.authenticated = True
			db.session.add(user)
			db.session.commit()
			
			# set remember=True to enable cookies to remember user
			fl.login_user(user, remember=True)
			flash('You were logged in', 'success')
			# return redirect(url_for('show_entries', username=user.username))
			
			# TODO: find out how to utilize this
			next_access = request.args.get('next')
			# next_is_valid should check if the user has valid
			# permission to access the `next` url
			if not has_permission(next_access):
				return abort(400)
			
			return redirect(next_access or url_for('upload_file', username=user.username))
	return render_template('login.html', form=form)


@login_required
@app.route('/<username>/logout')
def logout(username):
	check_authorization(username)
	user = current_user
	user.authenticated = False
	db.session.add(user)
	db.session.commit()
	fl.logout_user()
	flash('You were logged out', 'success')
	return redirect(url_for('login'))


@login_required
@app.route('/<username>/upload', methods=['GET', 'POST'])
def upload_file(username):
	check_authorization(username)
	form = FileForm()
	if form.validate_on_submit():
		file = form.file.data
		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			# TODO: make database model unique and handle database-errors instead of checking uniqueness
			if unique_filename(filename):
				# create folders for program
				folder_path = os.path.join(app.config['UPLOAD_FOLDER'], username, 'programs', filename.rsplit('.', 1)[0])
				try:
					os.mkdir(folder_path)
				except FileExistsError:
					pass
				create_folders(folder_path, ['results', 'old_results', 'plots', 'old_plots', 'activations', 'old_activations'])
				
				# save program in folder and create file meta
				file_path = os.path.join(folder_path, filename)
				file.save(file_path)
				file_meta = FileMeta(filename, date.today(), file_path, username)
				
				# create file tags
				for text in form.tags.data:
					file_meta.tags.append(get_existing_tag(text))
				
				# add file name as a tag automatically
				file_meta.tags.append(get_existing_tag(filename.split('.')[0]))
				
				# add file meta to database
				db.session.add(file_meta)
				db.session.commit()

				flash('File was successfully uploaded', 'success')
				return redirect(url_for('show_file', username=username, filename=filename))

			else:
				flash('Filename already exists', 'danger')
		else:
			flash('File type is not allowed', 'danger')
	return render_template('upload_file.html', form=form)


@login_required
@app.route('/<username>/uploads', methods=['GET', 'POST'])
def show_all_files(username):
	check_authorization(username)
	metas = FileMeta.query.filter_by(owner=username).all()
	search_form = SearchForm()
	if search_form.validate_on_submit():
		query = search_form.search.data
		if query:
			return redirect(url_for('search', username=username, query=query))
	return render_template('show_all_files.html', search_form=search_form, username=username, metas=metas)


@login_required
@app.route('/<username>/uploads/<filename>/code', methods=['GET', 'POST'])
def show_file_code(username, filename):
	meta = FileMeta.query.filter_by(filename=filename, owner=username).first()
	file = send_from_directory(os.path.join(app.config['UPLOAD_FOLDER'], username, 'programs', filename.rsplit('.', 1)[0]), filename)
	file.direct_passthrough = False
	content = str(file.data, 'utf-8')

	tag_form = TagForm()
	if tag_form.validate_on_submit():

		# if there is a tag to be deleted
		text = request.form.get('delete_tag')
		if text:
			tag = Tag.query.filter_by(text=text).first()
			meta.tags.remove(tag)
			db.session.commit()
			return redirect(url_for('show_file_code', username=username, filename=filename))

		# if not, tags should be added
		for text in tag_form.tags.data:
			meta.tags.append(get_existing_tag(text))
		db.session.commit()
		return redirect(url_for('show_file_code', username=username, filename=filename))
	return render_template('show_file_code.html', form=RunForm(), tag_form=TagForm(), username=username,
						   filename=filename, meta=meta, content=content)


@login_required
@app.route('/<username>/uploads/<filename>/visualization')
@app.route('/<username>/uploads/<filename>/visualization/<int:process_id>', methods=['GET', 'POST'])
def show_file_visualization(username, filename, process_id=0):
	meta = FileMeta.query.filter_by(filename=filename, owner=username).first()
	return render_template('show_file_visualization.html', username=username, filename=filename, meta=meta,
						   process_id=process_id)


@login_required
@app.route('/<username>/uploads/<filename>/history')
def show_file_history(username, filename):
	meta = FileMeta.query.filter_by(filename=filename, owner=username).first()
	return render_template('show_file_history.html', username=username, filename=filename, meta=meta)


@login_required
@app.route('/<username>/uploads/<filename>/visualization/plot/<int:process_id>')
def get_visualization_sources(username, filename, process_id):

	# get folder name from filename
	folder_name = filename.rsplit('.', 1)[0]

	# for plots
	plot_sources = []
	plot_message = 'File sources: '

	# get all static plot URLs
	plots = sorted(os.listdir(os.path.join(app.config['UPLOAD_FOLDER'], username, 'programs', folder_name, 'plots')))
	for plot in plots:
		plot_path = 'user_storage/%s/programs/%s/plots/%s' % (username, folder_name, plot)
		plot_sources.append(url_for('static', filename=plot_path))
		plot_message += 'static/%s, ' % plot_path

	if plots:
		# remove last comma
		plot_message = plot_message[:-2]
	else:
		# if not plots, relay in message
		plot_message = 'No plots produced yet'

	# for activations
	# each tuple contain sources for all activation images for one layer
	activation_tuples = []

	# get all static activation URLs and add to correct tuple
	activations = sorted(os.listdir(os.path.join(app.config['UPLOAD_FOLDER'], username, 'programs', folder_name, 'activations')))
	prev_activation_layer = ''
	for activation in activations:

		activation_path = 'user_storage/%s/programs/%s/activations/%s' % (username, folder_name, activation)

		# every activation starts with 'ln' and then a number, denoting layer number
		# if activation belongs to new layer, add new (layer title, act. path list)-tuple
		if activation[2] != prev_activation_layer:
			layer_title = 'Layer %s - %s' % (activation[2], activation.split('_', 2)[1].title())
			activation_tuples.append((layer_title, [url_for('static', filename=activation_path)]))
			prev_activation_layer = activation[2]
		# if not, append to act. path list for latest layer number
		else:
			activation_tuples[-1][1].append(url_for('static', filename=activation_path))

	# if writing process is alive, return 1
	# print('\nProcess ID: %d\n' % process_id)
	should_visualize = -1
	try:
		if app.config['processes'][process_id].is_alive():
			should_visualize = 1
	except KeyError:
		# process not found, consider it killed
		pass

	return jsonify(plot_sources=plot_sources, plot_message=plot_message, activation_tuples=activation_tuples, should_visualize=should_visualize)


@login_required
@app.route('/<username>/uploads/<filename>/run', methods=['POST'])
def run_upload(username, filename):
	meta = FileMeta.query.filter_by(filename=filename, owner=username).first()
	print('\n\nNew thread started for %s\n\n' % meta.path)
	
	# move results and plots if any exist
	if len(os.listdir(meta.path.replace(meta.filename, 'results'))) != 0:
		move_to_historical_folder(meta.path)
		
	# clear content on static URLs
	# clear the automatically added route for static
	# app.url_map._rules.clear()
	# app.url_map._rules_by_endpoint.clear()
	# # enable host matching and re-add the static route with the desired host
	# app.url_map.host_matching = True
	# app.add_url_rule(app.static_url_path + '/<path:filename>', endpoint='static', view_func=app.send_static_file)
	
	# shared boolean denoting if run_python_shell-process is writing
	shared_bool = Value('i', True)
	
	p = Process(target=run_python_shell, args=(meta.path, shared_bool))
	p.start()
	app.config['processes'][p.pid] = p

	p = Process(target=visualize_callback_output, args=(meta.path, meta.filename, shared_bool))
	p.start()
	app.config['processes'][p.pid] = p
	
	return redirect(url_for('show_file_visualization', username=username, filename=filename, process_id=p.pid))



@login_required
@app.route('/<username>/search_results/<query>')
def search(username, query):
	tags = query.split(" ")
	results = FileMeta.query.join(FileMeta.tags).filter(Tag.text.in_(tags))\
		.group_by(FileMeta).having(func.count(distinct(Tag.id)) == len(tags))
	return render_template('show_all_files.html', search_form=SearchForm(), username=username, metas=results)
