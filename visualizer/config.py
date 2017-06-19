import os
_basedir = os.path.abspath(os.path.dirname(__file__))

DEBUG = False
# temporary secret key
SECRET_KEY = 'thisissupersecretestkeyintheworld'

# entry to hold path to upload folder
UPLOAD_FOLDER = os.path.join(_basedir, 'user_storage')
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
				  ('saliency_maps', 'Saliency Maps'), ('deconvolutional_network', 'Deconvolutional Network'),
				  ('deep_visualization', 'Deep Visualization')]
# the intervals at which the bokeh visualizations should update at for the various visualizations
BOKEH_UPDATE_INTERVALS = {'training_progress': 200,
					'layer_activations': 10000,  # will take an especially long time to produce if network is large
					'saliency_maps': 5000,
					'deconvolutional_network': 5000,
					'deep_visualization': 5000}

# what command to use for running python
PYTHON = 'python3'
