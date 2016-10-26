import re
from flask import abort
from flask_login import current_user
from .models import User


ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}


def valid_username(username):
	# TODO: use regex operation re.search or re.match
	return True


def valid_password(password):
	# TODO: use regex operation re.search or re.match
	return True


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
