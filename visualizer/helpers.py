import re
import subprocess as sub
from flask import abort
from flask_login import current_user
from .models import User


ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'py'}


def valid_username(username):
	# checks if password only contains letters (capitalized or not), numbers and underscores
	if re.match('^[\w]*$', username):
		return True
	return False


def valid_password(password):
	# checks if password only contains letters (capitalized or not) and numbers,
	# and that it contains at least 1 capitalized letter,
	# and that it contains at least 1 non-capitalized letter,
	# and that it contains at least 2 numbers
	# if re.match('^[\w]*$', password) and \
	# 				len(re.findall('[A-Z]', password)) > 0 and \
	# 				len(re.findall('[a-z]', password)) > 0 and \
	# 				len(re.findall('[0-9]', password)) > 1:
	if True:
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
		

def get_form_errors(form):
	form_errors = []
	for field, errors in form.errors.items():
		for error in errors:
			form_errors.append('%s - %s' % (getattr(form, field).label.text, error))
	return form_errors


def run_python_shell(file_path):
	if file_path:
		# rel_path = 'temp_uploads\liste.py'
		# print('\n\nFile found at path: %s\n\n' % file.path)
		# print('\n\nTry using command: %s\n\n' % ('python ' + file.path))
		# cmd = 'python %s' % file.path
		# cmd = 'D: & cd ' + os.path.dirname(__file__) + ' & python ' + rel_path
		# cmd = '''cd ..'''
		# print('Try command:', cmd)
		sub.run('python ' + file_path)
		# sub.run('cd ' + os.path.dirname(__file__))
		# sub.run('python ' + rel_path)
	else:
		print('\n\nNo file found\n\n')
