from flask_wtf import FlaskForm
from wtforms import Field, StringField, PasswordField, TextAreaField, FileField, SubmitField
from wtforms.validators import DataRequired, EqualTo, Length
from wtforms.widgets import TextInput

USERNAME_MINIMUM_LENGTH = 5
USERNAME_MAXIMUM_LENGTH = 20
PASSWORD_MINIMUM_LENGTH = 8
	

class TagListField(Field):
	widget = TextInput()

	def _value(self):
		if self.data:
			return u', '.join(self.data)
		else:
			return u''

	def process_formdata(self, value_list):
		if value_list:
			self.data = [x.strip() for x in value_list[0].split(',')]
			self.data = filter(lambda tag: tag != '', self.data)
		else:
			self.data = []


class LoginForm(FlaskForm):
	username = StringField('Username', validators=[DataRequired()])
	password = PasswordField('Password', validators=[DataRequired()])
	submit = SubmitField('Login')


class CreateUserForm(FlaskForm):
	# username must have length in between the predetermined minimum and maximum length
	username = StringField('Username', validators=[DataRequired(),
												   Length(USERNAME_MINIMUM_LENGTH, USERNAME_MAXIMUM_LENGTH)])
	# password must be at least as long the predetermined minimum length,
	# and it must match the confirmation field
	password = PasswordField('Password', validators=[DataRequired(),
													 Length(PASSWORD_MINIMUM_LENGTH)])
	confirm = PasswordField('Confirm password', validators=[DataRequired(),
															EqualTo('password', u'Password does not match.')])
	submit = SubmitField('Create')

	
class EntryForm(FlaskForm):
	title = StringField('Title', validators=[DataRequired()])
	text = TextAreaField('Text', validators=[DataRequired()])
	submit = SubmitField('Share')
	

class FileForm(FlaskForm):
	file = FileField('Upload file', validators=[DataRequired('No selected file')])
	tags = TagListField('Tags')
	submit = SubmitField('Upload')


class RunForm(FlaskForm):
	run = SubmitField('Run')


class SearchForm(FlaskForm):
	search = StringField('Search')


class TagForm(FlaskForm):
	tags = TagListField('Tags')
