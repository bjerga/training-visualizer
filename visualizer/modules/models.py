from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Column, String, Integer, Boolean, ForeignKey, Table
from sqlalchemy.orm import relationship
from werkzeug.security import generate_password_hash, check_password_hash

# define database to use SQLAlchemy
db = SQLAlchemy()


# model for user
class User(db.Model):
	__tablename__ = 'User'
	username = Column(String(20), primary_key=True)
	
	# only save salted password, for security concerns
	salted_password = Column(String, nullable=False)
	
	# save user authentication (user login status)
	authenticated = Column(Boolean, default=False)
	
	# connections to other models
	files = relationship('FileMeta', backref='owned_by')
	
	def __init__(self, username, password, authenticated=False):
		self.username = username
		self.set_password(password)
		self.authenticated = authenticated
		
	def __repr__(self):
		return self.username
	
	# salt password
	def set_password(self, password):
		self.salted_password = generate_password_hash(password)
	
	# check salted password similarity
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


# association table for file tags
file_tags = Table('FileTags', db.Model.metadata,
					Column('tag_id', Integer, ForeignKey('Tag.id')),
					Column('file_id', Integer, ForeignKey('FileMeta.id')))


# model for tag
class Tag(db.Model):
	__tablename__ = 'Tag'
	id = Column(Integer, primary_key=True)
	text = Column(String(20), nullable=False, unique=True)

	def __init__(self, text):
		self.text = text

	def __repr__(self):
		return self.text


# model for information (meta) about file
class FileMeta(db.Model):
	__tablename__ = 'FileMeta'
	id = Column(Integer, primary_key=True)
	filename = Column(String(50), nullable=False)
	upload_date = Column(String(10), nullable=False)
	last_run_date = Column(String(16))
	path = Column(String(20), nullable=False)
	
	# connections to other models
	owner = Column(Integer, ForeignKey('User.username'), nullable=False)
	tags = relationship('Tag', secondary=file_tags, backref='files')
	
	def __init__(self, filename, upload_date, path, owner):
		self.filename = filename
		self.upload_date = upload_date
		self.last_run_date = None
		self.path = path
		self.owner = owner
	
	def __repr__(self):
		return self.filename
