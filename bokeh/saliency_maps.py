from os import listdir

from bokeh.io import curdoc
import numpy as np
from bokeh.layouts import layout
from bokeh.models import ColumnDataSource, Div, Paragraph

import sys
from os.path import dirname, join

from bokeh.plotting import figure
from PIL import Image

sys.path.insert(0, dirname(dirname(__file__)))

from visualizer.config import UPLOAD_FOLDER

document = curdoc()

args = document.session_context.request.arguments

# TODO: throw error if these are not provided
file = args['file'][0].decode('ascii')
user = args['user'][0].decode('ascii')

# find path for result data
results_path = join(UPLOAD_FOLDER, user, file, 'results')

# get original image
images_folder = join(UPLOAD_FOLDER, user, file, 'images')
image_name = listdir(images_folder)[-1]  # TODO: throw error here
original_image = np.array(Image.open(join(images_folder, image_name)))

# placeholder values for visualization
saliency_maps_data = np.zeros((1, 1, 1))

grid = []

div = Div(text="<h3>Visualization of the saliency maps</h3>", width=500)
grid.append([div])

p = Paragraph(text="There seems to be no visualizations produced yet.", width=500)
grid.append([p])

# flip image to display correctly in coordinate system
saliency_maps_source = ColumnDataSource(data=dict(image=[saliency_maps_data[::-1]]))

image_width = original_image.shape[0]
image_height = original_image.shape[1]

fig1 = figure(tools="box_zoom, reset, save", x_range=(0, image_width), y_range=(0, image_height))
fig1.title.text = "Original Image"
# add image, but we need to flip it to display correctly in coordinate system
fig1.image(image=[original_image[::-1]], x=0, y=0, dw=image_width, dh=image_height)
fig1.axis.visible = False
fig1.toolbar.logo = None

fig2 = figure(tools="box_zoom, reset, save", x_range=(0, image_width), y_range=(0, image_height))
fig2.title.text = "Saliency Map"
img = fig2.image(image='image', x=0, y=0, dw=image_width, dh=image_height, source=saliency_maps_source)
fig2.axis.visible = False
fig2.toolbar.logo = None

img.visible = False

grid.append([fig1, fig2])


def update_data():
	try:
		saliency_maps_data = np.load(join(results_path, 'saliency_maps.npy'))
		p.text = ""
		img.visible = True
	except FileNotFoundError:
		saliency_maps_data = np.zeros((1, 1, 1))

	# flip image to display correctly in coordinate system
	new_saliency_maps_data = dict(image=[saliency_maps_data[::-1]])
	saliency_maps_source.data = new_saliency_maps_data

update_data()

document.add_root(layout(grid))
document.add_periodic_callback(update_data, 5000)
