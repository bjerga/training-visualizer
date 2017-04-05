from bokeh.io import curdoc
import numpy as np
from bokeh.layouts import layout
from bokeh.models import ColumnDataSource, Div, Paragraph

import sys
from os.path import dirname, join

from bokeh.plotting import figure

sys.path.insert(0, dirname(dirname(__file__)))

from visualizer.config import UPLOAD_FOLDER

document = curdoc()

args = document.session_context.request.arguments

#TODO: throw error if these are not provided
file = args['file'][0].decode('ascii')
user = args['user'][0].decode('ascii')

results_path = join(UPLOAD_FOLDER, user, file, 'results')

saliency_maps_data = np.zeros((1, 1, 1))

grid = []

div = Div(text="<h3>Visualization of the saliency maps</h3>", width=500)
grid.append([div])

p = Paragraph(text="There seems to be no visualizations produced yet.", width=500)
grid.append([p])

saliency_maps_source = ColumnDataSource(data=dict(image=[saliency_maps_data]))

width = saliency_maps_source.data['image'][0].shape[0]
height = saliency_maps_source.data['image'][0].shape[1]

fig = figure(tools="box_zoom, reset, save", x_range=(0, width), y_range=(0, height))

img = fig.image(image='image', x=0, y=0, dw=width, dh=height, source=saliency_maps_source)
fig.axis.visible = False
fig.toolbar.logo = None

img.visible = False

grid.append([fig])


def update_data():
	try:
		saliency_maps_data = np.load(join(results_path, 'saliency_maps.npy'))
		p.text = ""
		img.visible = True
	except FileNotFoundError:
		saliency_maps_data = np.zeros((1, 1, 1))

	new_saliency_maps_data = dict(image=[saliency_maps_data])
	saliency_maps_source.data = new_saliency_maps_data

update_data()

document.add_root(layout(grid))
document.add_periodic_callback(update_data, 5000)
