from flask_wtf import FlaskForm
from wtforms import Form, StringField, PasswordField
from wtforms.validators import DataRequired


class UserForm(FlaskForm):
	username = StringField('Username', validators=[DataRequired()])
	password = PasswordField('Password', validators=[DataRequired()])

# TODO: create EntryForm?
