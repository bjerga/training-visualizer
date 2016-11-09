from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Column, String, Integer, Boolean, ForeignKey, Table
from sqlalchemy.orm import relationship
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()


class User(db.Model):
	__tablename__ = 'User'
	username = Column(String(20), primary_key=True)
	salted_password = Column(String, nullable=False)
	authenticated = Column(Boolean, default=False)
	files = relationship('FileMeta', backref='owned_by')
	entries = relationship('Entry', backref='owned_by')
	
	def __init__(self, username, password, autenticated=False):
		self.username = username
		self.set_password(password)
		self.authenticated = autenticated
		
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
	__tablename__ = 'Entry'
	id = Column(Integer, primary_key=True)
	title = Column(String(50), nullable=False)
	text = Column(String(120), nullable=False)
	owner = Column(Integer, ForeignKey('User.username'), nullable=False)
	
	def __init__(self, title, text, owner):
		self.title = title
		self.text = text
		self.owner = owner
	
	def __repr__(self):
		return self.title


file_tags = Table('FileTags', db.Model.metadata,
					Column('tag_id', Integer, ForeignKey('Tag.id')),
					Column('file_id', Integer, ForeignKey('FileMeta.id')))


class Tag(db.Model):
	__tablename__ = 'Tag'
	id = Column(Integer, primary_key=True)
	text = Column(String(20), nullable=False)

	def __init__(self, text):
		self.text = text

	def __repr__(self):
		return self.text


class FileMeta(db.Model):
	__tablename__ = 'FileMeta'
	id = Column(Integer, primary_key=True)
	filename = Column(String(50), nullable=False)
	upload_date = Column(String(10), nullable=False)
	path = Column(String(20), nullable=False)
	owner = Column(Integer, ForeignKey('User.username'), nullable=False)
	tags = relationship('Tag', secondary=file_tags, backref='files')
	
	def __init__(self, filename, upload_date, path, owner):
		self.filename = filename
		self.upload_date = upload_date
		self.path = path
		self.owner = owner
	
	def __repr__(self):
		return self.filename

