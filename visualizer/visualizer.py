import datetime
from flask import Flask, request, session, g, redirect, url_for, abort, render_template, flash, send_from_directory
from werkzeug.utils import secure_filename
import flask_login as fl
from flask_login import login_required
from .helpers import *
from .models import *
from .forms import *


# UPLOAD_FOLDER = 'temp_uploads'
# ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

# Create application
app = Flask(__name__)
app.config.from_object(__name__)

# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///visualizer.db'
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
	error = None
	form = UserForm()
	if form.validate_on_submit():
		if not valid_username(form.username.data):
			error = 'Username is invalid'
		elif not unique_username(form.username.data):
			error = 'Username is already taken'
		elif not valid_password(form.password.data):
			error = 'Password is invalid'
		else:
			db.session.add(User(form.username.data, form.password.data))
			db.session.commit()
			flash('User successfully created. Try logging in!')
			return redirect(url_for('login'))
	return render_template('create_user.html', form=form, error=error,
						   username_min_length=USERNAME_MINIMUM_LENGTH,
						   username_max_length=USERNAME_MAXIMUM_LENGTH,
						   password_min_length=PASSWORD_MINIMUM_LENGTH)


@app.route('/', methods=['GET', 'POST'])
def login():
	error = None
	form = UserForm()
	if form.validate_on_submit():
		user = User.query.filter_by(username=form.username.data).first()
		# user = load_user(form.username.data)
		if not user:
			error = 'Invalid username'
		elif not user.check_password(form.password.data):
			error = 'Invalid password'
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
	return render_template('login.html', form=form, error=error)


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
	return render_template('show_entries.html', username=username, entries=entries)


@login_required
@app.route('/<username>/entries', methods=['POST'])
def add_entry(username):
	check_authorization(username)
	db.session.add(Entry(request.form['title'], request.form['text'], username))
	db.session.commit()
	return redirect(url_for('show_entries', username=username))


@login_required
@app.route('/<username>/upload', methods=['GET', 'POST'])
def upload_file(username):
	check_authorization(username)

	if request.method == 'POST':
		# check if the post request has the file part
		if 'file' not in request.files:
			flash('No file part')
			return redirect(request.url)
		file = request.files['file']
		# if user does not select file, browser also
		# submit a empty part without filename
		if file.filename == '':
			flash('No selected file')
			return redirect(request.url)
		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			# file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			db.session.add(File(filename, datetime.date.today(), 'replace with path', username))
			db.session.commit()
			flash('File was successfully uploaded')
			return redirect(url_for('show_file', username=username, filename=filename))
	return render_template('upload.html')


@login_required
@app.route('/<username>/uploads/all')
def show_all_files(username):
	check_authorization(username)
	files = File.query.filter_by(owner=username).all()
	return render_template('show_all_files.html', files=files)


@login_required
@app.route('/<username>/uploads/<filename>')
def show_file(username, filename):
	check_authorization(username)
	file = File.query.filter_by(name=filename, owner=username).first()
	return render_template('show_file.html', file=file)
