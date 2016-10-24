import os, datetime
import sqlite3
from flask import Flask, request, session, g, redirect, url_for, abort, render_template, flash, send_from_directory
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy

UPLOAD_FOLDER = 'temp_uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}


# Create application
app = Flask(__name__)
app.config.from_object(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///visualizer.db'
db = SQLAlchemy(app)


# Load default config and override config from an environment variable
app.config.update(dict(
	SECRET_KEY='development key',
	USERNAME='admin',
	PASSWORD='admin'
))
app.config.from_envvar('VISUALIZER_SETTINGS', silent=True)


class Entry(db.Model):
	id = db.Column(db.Integer, primary_key=True)
	title = db.Column(db.String(50), nullable=False)
	text = db.Column(db.String(120), nullable=False)

	def __init__(self, title, text):
		self.title = title
		self.text = text

	def __repr__(self):
		return self.title


class File(db.Model):
	id = db.Column(db.Integer, primary_key=True)
	name = db.Column(db.String(50), nullable=False)
	upload_date = db.Column(db.String(10), nullable=False)
	content = db.Column(db.String(20), nullable=False)

	def __init__(self, name, upload_date, content):
		self.name = name
		self.upload_date = upload_date
		self.content = content

	def __repr__(self):
		return self.name


@app.cli.command('initdb')
def initdb_command():
	"""Initializes the database."""
	db.create_all()
	print('Initialized the database')


@app.route('/entries')
def show_entries():
	entries = Entry.query.all()
	return render_template('show_entries.html', entries=entries)


@app.route('/entries', methods=['POST'])
def add_entry():
	if not session.get('logged_in'):
		abort(401)
	db.session.add(Entry(request.form['title'], request.form['text']))
	db.session.commit()
	return redirect(url_for('show_entries'))


@app.route('/', methods=['GET', 'POST'])
def login():
	error = None
	if request.method == 'POST':
		if request.form['username'] != app.config['USERNAME']:
			error = 'Invalid username'
		elif request.form['password'] != app.config['PASSWORD']:
			error = 'Invalid password'
		else:
			session['logged_in'] = True
			flash('You were logged in')
			return redirect(url_for('show_entries'))
	return render_template('login.html', error=error)


@app.route('/logout')
def logout():
	session.pop('logged_in', None)
	flash('You were logged out')
	return redirect(url_for('show_entries'))


def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
	if not session.get('logged_in'):
		abort(401)

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
			flash('New entry was successfully posted')
			filename = secure_filename(file.filename)
			# file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			db.session.add(File(filename, datetime.date.today(), 'replace with content'))
			db.session.commit()
			flash('File was successfully uploaded')
			return redirect(url_for('show_file', filename=filename))
	return render_template('upload.html')


@app.route('/uploads/all')
def show_all_files():
	files = File.query.all()
	return render_template('show_all_files.html', files=files)


@app.route('/uploads/<filename>')
def show_file(filename):
	file = File.query.filter_by(name=filename).first()
	return render_template('show_file.html', file=file)
