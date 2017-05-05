import os
_basedir = os.path.abspath(os.path.dirname(__file__))

DEBUG = False
# temporary secret key
SECRET_KEY = 'thisissupersecretestkeyintheworld'

# entry to hold path to upload folder
UPLOAD_FOLDER = os.path.join(_basedir, 'static', 'user_storage')
# number of lines to print of the script output
NO_OF_OUTPUT_LINES = 50

# entry to hold database connection
SQLALCHEMY_DATABASE_URI = 'sqlite:///visualizer.db'
# set to disable notifications of overhead when running
SQLALCHEMY_TRACK_MODIFICATIONS = False

# URL to bokeh server
BOKEH_SERVER = 'http://localhost:5006'
# the different visualization techniques available
VISUALIZATIONS = [('training_progress', 'Training Progress'), ('layer_activations', 'Layer Activations'),
				  ('saliency_maps', 'Saliency Maps'), ('deconvolution_network', 'Deconvolution Network'),
				  ('deep_visualization', 'Deep Visualization')]

# what command to use for running python
PYTHON = 'python3'
