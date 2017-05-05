from os import listdir

from bokeh.io import curdoc
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource, Div, Paragraph, Column, Range1d

from os.path import join

import numpy as np

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

# used to determine if the data source should be created
# this is needed to avoid error when the script is just started
create_source = True

div = Div(text="<h3>Deconvolution Network</h3>", width=500)
layout = Column(children=[div])

p = Paragraph(text="There seems to be no visualizations produced yet.", width=500)
layout.children.append(p)

image_height = original_image.shape[0]
image_width = original_image.shape[1]


# convert image from 3-dimensional to 2-dimensional
def process_rgba_image(img):
	if img.shape[2] == 3:
		img = np.dstack([img, np.ones(img.shape[:2], np.uint8) * 255])
	img = np.squeeze(img.view(np.uint32))
	return img


# create plot for the original image
img_fig = figure(tools="box_zoom, reset, save, pan")
img_fig.x_range = Range1d(0, image_width, bounds=(0, image_width))
img_fig.y_range = Range1d(0, image_height, bounds=(0, image_height))
img_fig.title.text = "Original Image"
img_fig.outline_line_color = "black"
img_fig.outline_line_width = 3
img_fig.axis.visible = False
img_fig.plot_width = 250
img_fig.plot_height = 250

# check if image is rgb or grayscale
if original_image.ndim > 2:
	img_fig.image_rgba(image=[process_rgba_image(original_image)[::-1]], x=0, y=0, dw=image_width, dh=image_height)
else:
	img_fig.image(image=[original_image[::-1]], x=0, y=0, dw=image_width, dh=image_height)

img_grid = gridplot([img_fig], ncols=1, toolbar_options=dict(logo=None))
layout.children.append(img_grid)

deconvolution_source = ColumnDataSource(data=dict())


def fill_data_source(deconvolution_data):

	p.text = "Visualizations are being produced..."
	figures = []

	# loop through feature maps
	for i in range(len(deconvolution_data)):

		layer_name = deconvolution_data[i][0]
		array = deconvolution_data[i][1]

		name = "{}_{}".format(layer_name, i)
		title = "Feature map #{} in {}".format(i, layer_name)

		# process if rgb image
		rgb = False
		if array.ndim > 2:
			array = process_rgba_image(array)
			rgb = True

		# add image to the data source
		deconvolution_source.add([array[::-1]], name=name)

		# create plots for feature map
		fig = create_figure(rgb, deconvolution_source, name, title, array.shape[0], array.shape[1])
		fig.x_range = img_fig.x_range
		fig.y_range = img_fig.y_range
		figures.append(fig)

	# make a grid of the feature maps
	grid = gridplot(figures, ncols=4, toolbar_options=dict(logo=None))
	layout.children.append(grid)
	p.text = ""


def create_figure(rgb, source, image_name, title, dw, dh, tools="box_zoom, reset, save, pan"):
	# link range to plot of original image
	fig = figure(tools=tools, plot_width=250, plot_height=250, x_range=img_fig.x_range, y_range=img_fig.y_range)
	fig.title.text = title

	if rgb:
		fig.image_rgba(image=image_name, x=0, y=0, dw=dw, dh=dh, source=source)
	else:
		fig.image(image=image_name, x=0, y=0, dw=dw, dh=dh, source=source)
	fig.outline_line_color = "black"
	fig.outline_line_width = 3
	fig.axis.visible = False
	return fig


def update_data():
	global create_source
	try:
		with open(join(results_path, 'deconvolution_network.pickle'), 'rb') as f:
			deconvolution_data = pickle.load(f)

		# if it is the first time data is detected, we need to fill the data source with the images
		if create_source:
			# temporary remove callback to make sure the function is not being called while creating the visualizations
			document.remove_periodic_callback(update_data)
			fill_data_source(deconvolution_data)
			create_source = False
			document.add_periodic_callback(update_data, 5000)
		# if not, we can simply update the data
		else:
			for i in range(len(deconvolution_data)):
				layer_name = deconvolution_data[i][0]
				array = deconvolution_data[i][1]
				name = "{}_{}".format(layer_name, i)

				# process if rgb image
				if array.ndim > 2:
					array = process_rgba_image(array)

				deconvolution_source.data[name] = [array[::-1]]

	except FileNotFoundError:
		# this means deconvolution data has not been created yet, skip visualization
		return
	except EOFError:
		# this means deconvolution data has been created, but is empty, skip visualization
		return

document.add_root(layout)
document.add_periodic_callback(update_data, 5000)
update_data()
