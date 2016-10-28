import datetime, os, threading
from flask import Flask, request, session, g, redirect, url_for, abort, render_template, flash, send_from_directory
from werkzeug.utils import secure_filename
import flask_login as fl
from flask_login import login_required
from .helpers import *
from .models import *
from .forms import *


# Create application
app = Flask(__name__)
app.config.from_object(__name__)

UPLOAD_FOLDER = 'temp_uploads'
threads = []

# app.config['UPLOAD_FOLDER'] = os.path.join('visualizer', UPLOAD_FOLDER)
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
app.config.update(dict(
	SECRET_KEY='thisissupersecret',
))
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
			db.session.add(User(form.username.data, form.password.data))
			db.session.commit()
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
			path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
			file.save(path)
			db.session.add(FileMeta(filename, datetime.date.today(), path, username))
			db.session.commit()
			meta_id = FileMeta.query.filter_by(filename=filename, owner=username).first().id
			for tag in form.tags.data:
				db.session.add(Tag(meta_id, tag))
			db.session.commit()
			flash('File was successfully stored in database')
			return redirect(url_for('show_file', username=username, filename=filename))
	return render_template('upload.html', form=form, errors=get_form_errors(form))


@login_required
@app.route('/<username>/uploads')
def show_all_files(username):
	check_authorization(username)
	metas = FileMeta.query.filter_by(owner=username).all()
	return render_template('show_all_files.html', username=username, metas=metas)


@login_required
@app.route('/<username>/uploads/<filename>')
def show_file(username, filename):
	check_authorization(username)
	meta = FileMeta.query.filter_by(filename=filename, owner=username).first()
	file = send_from_directory(UPLOAD_FOLDER, filename)
	file.direct_passthrough = False
	content = str(file.data, 'utf-8')
	return render_template('show_file.html', form=RunForm(), username=username, filename=filename, meta=meta, content=content)


@login_required
@app.route('/<username>/uploads/<filename>', methods=['POST'])
def run_upload(username, filename):
	meta = FileMeta.query.filter_by(filename=filename, owner=username).first()
	print('\n\nNew thread started for %s\n\n' % meta.path)
	t = threading.Thread(target=run_python_shell, args=(meta.path,))
	threads.append(t)
	t.start()
	return redirect(url_for('show_file', username=username, filename=filename))
