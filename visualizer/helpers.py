import re
from flask import abort
from flask_login import current_user
from .models import User


ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
USERNAME_MINIMUM_LENGTH = 6
USERNAME_MAXIMUM_LENGTH = 20
PASSWORD_MINIMUM_LENGTH = 8


def valid_username(username):
	# checks if password only contains letters (capitalized or not), numbers and underscores,
	# and that it is has a length in between the predetermined minimum and maximum length
	if re.match('^[\w]*$', username) and USERNAME_MINIMUM_LENGTH <= len(username) <= USERNAME_MAXIMUM_LENGTH:
		return True
	return False


def valid_password(password):
	# checks if password only contains letters (capitalized or not) and numbers,
	# and that it is at least as long the predetermined minimum length,
	# and that it contains at least 1 capitalized letter,
	# and that it contains at least 1 non-capitalized letter,
	# and that it contains at least 2 numbers
	if re.match('^[\w]*$', password) and \
					len(password) >= PASSWORD_MINIMUM_LENGTH and \
					len(re.findall('[A-Z]', password)) > 0 and \
					len(re.findall('[a-z]', password)) > 0 and \
					len(re.findall('[0-9]', password)) > 1:
		return True
	return False


def unique_username(username):
	if User.query.filter_by(username=username).first():
		return False
	return True


def has_permission(next_access):
	# TODO: implement
	return True


def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


# TODO: authorization could be improved
def check_authorization(username):
	if username != str(current_user):
		abort(401)
