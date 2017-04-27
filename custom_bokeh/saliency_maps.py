from os import listdir

from bokeh.io import curdoc
import numpy as np
from bokeh.layouts import layout
from bokeh.models import ColumnDataSource, Div, Paragraph, Range1d

from os.path import join

from bokeh.plotting import figure
from PIL import Image
import pickle

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

grid = []

div = Div(text="<h3>Visualization of the saliency maps</h3>", width=500)
grid.append([div])

p = Paragraph(text="There seems to be no visualizations produced yet.", width=500)
grid.append([p])

# flip image to display correctly in coordinate system with placeholder
saliency_maps_source = ColumnDataSource(data=dict(image=[np.zeros((1, 1, 1))]))

image_height = original_image.shape[0]
image_width = original_image.shape[1]


def create_figure(title, x_range, y_range, tools="box_zoom, reset, save, pan"):
	fig = figure(tools=tools, x_range=x_range, y_range=y_range)
	fig.title.text = title
	fig.axis.visible = False
	fig.toolbar.logo = None
	return fig


# create figures and add to grid
range_x = Range1d(0, image_width, bounds=(0, image_width))
range_y = Range1d(0, image_height, bounds=(0, image_height))

fig1 = create_figure('Original Image', range_x, range_y)
fig1.image(image=[original_image[::-1]], x=0, y=0, dw=image_width, dh=image_height)
fig2 = create_figure('Saliency Map', fig1.x_range, fig1.y_range)
fig2.image(image='image', x=0, y=0, dw=image_width, dh=image_height, source=saliency_maps_source)

grid.append([fig1, fig2])


def update_data():
	try:
		with open(join(results_path, 'saliency_maps.pickle'), 'rb') as f:
			saliency_maps_data = pickle.load(f)
		p.text = ""
		# flip image to display correctly in coordinate system
		new_saliency_maps_data = dict(image=[saliency_maps_data[::-1]])
		saliency_maps_source.data = new_saliency_maps_data
	except FileNotFoundError:
		# if no visualization has been produced yet, simply skip visualization
		return

update_data()

document.add_root(layout(grid))
document.add_periodic_callback(update_data, 5000)
