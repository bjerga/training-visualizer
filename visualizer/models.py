from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Column, String, Integer, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()


class User(db.Model):
	username = Column(String(20), primary_key=True)
	salted_password = Column(String, nullable=False)
	authenticated = Column(Boolean, default=False)
	files = relationship('File', backref='owned_by')
	entries = relationship('Entry', backref='owned_by')
	
	def __init__(self, username, password):
		self.username = username
		self.set_password(password)
	
	def __repr__(self):
		return self.username
	
	def set_password(self, password):
		self.salted_password = generate_password_hash(password)
	
	def check_password(self, password):
		return check_password_hash(self.salted_password, password)
	
	# next four methods are needed for flask_login
	def get_id(self):
		return self.username
	
	def is_authenticated(self):
		return self.authenticated
	
	def is_active(self):
		# all users are active
		return True
	
	def is_anonymous(self):
		# no users are anonymous
		return False


class Entry(db.Model):
	id = Column(Integer, primary_key=True)
	title = Column(String(50), nullable=False)
	text = Column(String(120), nullable=False)
	owner = Column(Integer, ForeignKey('user.username'), nullable=False)
	
	def __init__(self, title, text, owner):
		self.title = title
		self.text = text
		self.owner = owner
	
	def __repr__(self):
		return self.title


class File(db.Model):
	id = Column(Integer, primary_key=True)
	name = Column(String(50), nullable=False)
	upload_date = Column(String(10), nullable=False)
	path = Column(String(20), nullable=False)
	owner = Column(Integer, ForeignKey('user.username'), nullable=False)
	
	def __init__(self, name, upload_date, path, owner):
		self.name = name
		self.upload_date = upload_date
		self.path = path
		self.owner = owner
	
	def __repr__(self):
		return self.name
