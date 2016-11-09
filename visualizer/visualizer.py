import os
from datetime import date
from multiprocessing import Process, Value

import flask_login as fl
from flask import Flask, request, redirect, url_for, render_template, flash, send_from_directory
from flask_login import login_required
from werkzeug.utils import secure_filename

from visualizer.modules.helpers import *
from visualizer.modules.models import *
from visualizer.modules.forms import *

# Create application
app = Flask(__name__)
app.config.from_object(__name__)
app.secret_key = 'thisissupersecretestkeyintheworld'

UPLOAD_FOLDER = 'user_storage'
processes = []

app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), UPLOAD_FOLDER)
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
	db.drop_all()
	db.create_all()
	print('Initialized the database')


@app.route('/create_user', methods=['GET', 'POST'])
def create_user():
	form = CreateUserForm()
	if form.validate_on_submit():
		errors = []
		if not valid_username(form.username.data):
			errors.append('Username does not match requirements')
		elif not unique_username(form.username.data):
			errors.append('Username is already taken')
		if not valid_password(form.password.data):
			errors.append('Password does not match requirements')
		if len(errors) == 0:
			# add user to database
			db.session.add(User(form.username.data, form.password.data))
			db.session.commit()
			
			# create folders for user to save data in
			try:
				os.mkdir(app.config['UPLOAD_FOLDER'] + '/' + form.username.data)
			except FileExistsError:
				pass
			try:
				os.mkdir(app.config['UPLOAD_FOLDER'] + '/' + form.username.data + '/programs')
			except FileExistsError:
				pass
			try:
				os.mkdir(app.config['UPLOAD_FOLDER'] + '/' + form.username.data + '/plots')
			except FileExistsError:
				pass
			
			flash('User successfully created. Try logging in!')
			return redirect(url_for('login'))
	else:
		errors = get_form_errors(form)
	return render_template('create_user.html', form=form, errors=errors,
						   username_min_length=USERNAME_MINIMUM_LENGTH,
						   username_max_length=USERNAME_MAXIMUM_LENGTH,
						   password_min_length=PASSWORD_MINIMUM_LENGTH)


@app.route('/', methods=['GET', 'POST'])
def login():
	form = LoginForm()
	if form.validate_on_submit():
		user = User.query.filter_by(username=form.username.data).first()
		# user = load_user(form.username.data)
		if not user:
			errors = ['Invalid username']
		elif not user.check_password(form.password.data):
			errors = ['Invalid password']
		else:
			user.authenticated = True
			db.session.add(user)
			db.session.commit()
			
			# set remember=True to enable cookies to remember user
			fl.login_user(user, remember=True)
			flash('You were logged in')
			# return redirect(url_for('show_entries', username=user.username))
			
			# TODO: find out how to utilize this
			next_access = request.args.get('next')
			# next_is_valid should check if the user has valid
			# permission to access the `next` url
			if not has_permission(next_access):
				return abort(400)
			
			return redirect(next_access or url_for('show_entries', username=user.username))
	else:
		errors = get_form_errors(form)
	return render_template('login.html', form=form, errors=errors)


@login_required
@app.route('/<username>/logout')
def logout(username):
	check_authorization(username)
	user = current_user
	user.authenticated = False
	db.session.add(user)
	db.session.commit()
	fl.logout_user()
	flash('You were logged out')
	return redirect(url_for('login'))


@login_required
@app.route('/<username>/entries')
def show_entries(username):
	check_authorization(username)
	entries = Entry.query.filter_by(owner=username).all()
	return render_template('show_entries.html', form=EntryForm(), username=username, entries=entries)


@login_required
@app.route('/<username>/entries', methods=['POST'])
def add_entry(username):
	check_authorization(username)
	form = EntryForm()
	if form.validate_on_submit():
		db.session.add(Entry(form.title.data, form.text.data, username))
		db.session.commit()
	else:
		flash('Message was invalid')
	return redirect(url_for('show_entries', form=form, username=username))


@login_required
@app.route('/<username>/upload', methods=['GET', 'POST'])
def upload_file(username):
	check_authorization(username)
	form = FileForm()
	if form.validate_on_submit():
		file = form.file.data
		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			path = os.path.join(app.config['UPLOAD_FOLDER'], username, 'programs', filename)
			file.save(path)
			file_meta = FileMeta(filename, date.today(), path, username)
			for tag in form.tags.data:
				file_meta.tags.append(Tag(tag))
			file_meta.tags.append(Tag(filename.split('.')[0]))
			db.session.add(file_meta)
			db.session.commit()
			flash('File was successfully stored in database')
			return redirect(url_for('show_file', username=username, filename=filename))
	return render_template('upload_file.html', form=form, errors=get_form_errors(form))


@login_required
@app.route('/<username>/uploads', methods=['GET', 'POST'])
def show_all_files(username):
	check_authorization(username)
	metas = FileMeta.query.filter_by(owner=username).all()
	search_form = SearchForm()
	if search_form.validate_on_submit():
		query = search_form.search.data
		return redirect(url_for('search', username=username, query=query))
	return render_template('show_all_files.html', search_form=search_form, username=username, metas=metas)


@login_required
@app.route('/<username>/uploads/<filename>', methods=['GET', 'POST'])
def show_file(username, filename):
	check_authorization(username)
	meta = FileMeta.query.filter_by(filename=filename, owner=username).first()
	file = send_from_directory(os.path.join(app.config['UPLOAD_FOLDER'], username, 'programs'), filename)
	file.direct_passthrough = False
	content = str(file.data, 'utf-8')
	tag_form = TagForm()
	if tag_form.validate_on_submit():
		for tag in tag_form.tags.data:
				meta.tags.append(Tag(tag))
		db.session.commit()
		return redirect(url_for('show_file', username=username, filename=filename))
	return render_template('show_file.html', form=RunForm(), username=username, filename=filename, meta=meta, content=content, tag_form=TagForm())


@login_required
@app.route('/<username>/uploads/<filename>', methods=['POST'])
def run_upload(username, filename):
	meta = FileMeta.query.filter_by(filename=filename, owner=username).first()
	print('\n\nNew thread started for %s\n\n' % meta.path)
	
	# shared boolean denoting if run_python_shell-process is writing
	shared_bool = Value('i', True)
	
	p = Process(target=run_python_shell, args=(meta.path, shared_bool))
	processes.append(p)
	p.start()

	p = Process(target=plot_accuracy_error, args=(meta.path, shared_bool))
	processes.append(p)
	p.start()
	
	return redirect(url_for('show_file', username=username, filename=filename))


@login_required
@app.route('/<username>/search_results/<query>')
def search(username, query):
	results = Tag.query.filter_by(text=query).all()
	search_form = SearchForm()
	metas = set()
	for result in results:
		metas |= set(result.files)
	return render_template('show_all_files.html', search_form=search_form, username=username, metas=metas)


@login_required
@app.route('/<username>/uploads/<filename>/remove_tag/<tag_id>', methods=['POST'])
def remove_tag(username, filename, tag_id):
	meta = FileMeta.query.filter_by(filename=filename, owner=username).first()
	tag = Tag.query.get(tag_id)
	meta.tags.remove(tag)
	db.session.commit()
	return redirect(url_for('show_file', username=username, filename=filename))

