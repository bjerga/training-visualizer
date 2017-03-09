from os.path import join, dirname

DEBUG = True
# temporary secret key
SECRET_KEY = 'thisissupersecretestkeyintheworld'

# entry to hold path to upload folder
UPLOAD_FOLDER = join(dirname(__file__), 'static', 'user_storage')

# entry to hold database connection
SQLALCHEMY_DATABASE_URI = 'sqlite:///visualizer.db'
# set to disable notifications of overhead when running
SQLALCHEMY_TRACK_MODIFICATIONS = False

# URL to bokeh server
BOKEH_SERVER='http://localhost:5006'


