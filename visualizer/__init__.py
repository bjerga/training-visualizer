from flask import Flask
# Create application and get config
app = Flask(__name__)
app.config.from_object('visualizer.config')

from os.path import dirname
import sys
sys.path.insert(0, dirname(__file__))

import visualizer.visualizer
