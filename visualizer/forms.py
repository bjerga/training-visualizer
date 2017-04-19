from flask_wtf import FlaskForm
from wtforms import Field, StringField, PasswordField, TextAreaField, FileField, SubmitField, SelectField, RadioField
from wtforms.validators import DataRequired, EqualTo, Length, Regexp
from wtforms.widgets import TextInput

# determined values of username and password length
USERNAME_MINIMUM_LENGTH = 5
USERNAME_MAXIMUM_LENGTH = 20
PASSWORD_MINIMUM_LENGTH = 8

# options for visualization dropdown, value is the url corresponding to the visualization
visualization_choices = [('saliency_maps', 'Saliency Maps'), ('layer_activations', 'Layer Activations')]


# custom field for tag form
class TagListField(Field):
	widget = TextInput()

	def _value(self):
		if self.data:
			return u', '.join(self.data)
		else:
			return u''

	def process_formdata(self, value_list):
		# if tags specified
		if value_list:
			# split text into separate tags
			self.data = [x.strip() for x in value_list[0].split()]
			
			# remove any empty tags
			self.data = filter(lambda tag: tag != '', self.data)
		else:
			# if no tags, set to empty list
			self.data = []


# form for login, require text in username- and password-field
class LoginForm(FlaskForm):
	username = StringField('Username', validators=[DataRequired()])
	password = PasswordField('Password', validators=[DataRequired()])
	submit = SubmitField('Login')


# form for creating a user
class CreateUserForm(FlaskForm):
	# username field requires:
	# - data in the field
	# - only alphanumerical characters, or underscores
	# - length in between the predetermined minimum and maximum length
	username = StringField('Username', validators=[DataRequired(),
												   Regexp('^\w+$', message=u'Username must only contain letters, '
																		   u'numbers and underscores'),
												   Length(USERNAME_MINIMUM_LENGTH, USERNAME_MAXIMUM_LENGTH)])
	# password field requires:
	# - data in the field
	# - only alphanumerical characters
	# - password must be at least as long the predetermined minimum length
	password = PasswordField('Password', validators=[DataRequired(),
													 Regexp('^[a-zA-Z0-9]+$', message=u'Passord must only contain letters '
																			 u'and numbers'),
													 Length(PASSWORD_MINIMUM_LENGTH)])
	
	# confirm field requires:
	# - data in the field
	# - text must match the password field
	confirm = PasswordField('Confirm password', validators=[DataRequired(),
															EqualTo('password', u'Password does not match.')])
	submit = SubmitField('Create')


# form for uploading files
class FileForm(FlaskForm):
	# file is required
	file = FileField('Upload file', validators=[DataRequired('No selected file')])
	
	# tags are optional
	tags = TagListField('Tags')
	
	submit = SubmitField('Upload')


# form for run button in file view
class RunForm(FlaskForm):
	image = FileField('Upload image', validators=[])
	run = SubmitField('Run')


# form for search bar in file list view
class SearchForm(FlaskForm):
	search = StringField('Search')


# form for tags, uses custom tag field
class TagForm(FlaskForm):
	tags = TagListField('Tags')


class VisualizationForm(FlaskForm):
	visualization = SelectField(label="Select a visualization technique:", choices=visualization_choices)
