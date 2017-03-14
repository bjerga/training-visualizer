# Import flask
from flask import Flask

# Import SQLAlchemy
from flask_sqlalchemy import SQLAlchemy

# Import LoginManager
from flask_login import LoginManager

# Define the WSGI application object
app = Flask(__name__)

# Configurations
app.config.from_object('visualizer.config')

# Define the database object which is imported by modules
db = SQLAlchemy(app)

# Build the database:
# This will create the database file using SQLAlchemy
db.create_all()

# Initialize login manager
login_manager = LoginManager()
login_manager.init_app(app)

# Import views
import visualizer.views
