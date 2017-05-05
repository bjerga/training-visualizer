from os import listdir

from bokeh.io import curdoc
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource, Div, Paragraph, Column, Range1d

from os.path import join

from bokeh.plotting import figure
from PIL import Image
import pickle

from visualizer.config import UPLOAD_FOLDER
from custom_bokeh.helpers import *

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
orig_img = np.array(Image.open(join(images_folder, image_name)))

# used to determine if the data source should be created
# this is needed to avoid error when the script is just started
create_source = True
deconvolution_source = ColumnDataSource(data=dict())

div = Div(text="<h3>Deconvolution Network</h3>", width=500)
layout = Column(children=[div])

p = Paragraph(text="There seems to be no visualizations produced yet.", width=500)
layout.children.append(p)

image_height = orig_img.shape[0]
image_width = orig_img.shape[1]


def is_grayscale(img):
	if img.ndim == 2:
		return True
	if img.shape[2] == 1:
		return True
	return False


def add_image_from_array(fig, img):
	if is_grayscale(img):
		img = process_image_dim(img)
		fig.image(image=[img[::-1]], x=0, y=0, dw=img.shape[0], dh=img.shape[1])
	else:
		img = process_image_dim(img)
		fig.image_rgba(image=[img[::-1]], x=0, y=0, dw=img.shape[0], dh=img.shape[1])


def add_image_from_source(fig, source, img, img_name):
	if is_grayscale(img):
		img = process_image_dim(img)
		fig.image(image=img_name, x=0, y=0, dw=img.shape[0], dh=img.shape[1], source=source)
	else:
		img = process_image_dim(img)
		fig.image_rgba(image=img_name, x=0, y=0, dw=img.shape[0], dh=img.shape[1], source=source)
	source.add([img[::-1]], name=img_name)


# create plot for the original image
img_fig = figure(title="Original Image", plot_width=250, plot_height=250, tools="box_zoom, reset, save, pan",
					outline_line_color="black", outline_line_width=3)
img_fig.x_range = Range1d(0, image_width, bounds=(0, image_width))
img_fig.y_range = Range1d(0, image_height, bounds=(0, image_height))
img_fig.axis.visible = False


add_image_from_array(img_fig, orig_img)

img_grid = gridplot([img_fig], ncols=1, toolbar_options=dict(logo=None))

layout.children.append(img_grid)


def fill_data_source(deconvolution_data):
	p.text = "Visualizations are being produced..."
	figures = []

	# loop through feature maps
	for i in range(len(deconvolution_data)):

		layer_name = deconvolution_data[i][0]
		array = deconvolution_data[i][1]

		name = "{}_{}".format(layer_name, i)
		title = "Feature map #{} in {}".format(i, layer_name)

		fig = figure(title=title, tools="box_zoom, reset, save, pan", plot_width=250, plot_height=250,
						outline_line_color="black", outline_line_width=3, x_range=img_fig.x_range, y_range=img_fig.y_range)
		fig.axis.visible = False
		add_image_from_source(fig, deconvolution_source, array, name)

		figures.append(fig)

	# make a grid of the feature maps
	grid = gridplot(figures, ncols=4, toolbar_options=dict(logo=None))
	layout.children.append(grid)
	p.text = ""


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

				img = process_image_dim(array)
				deconvolution_source.data[name] = [img[::-1]]

	except FileNotFoundError:
		# this means deconvolution data has not been created yet, skip visualization
		return
	except EOFError:
		# this means deconvolution data has been created, but is empty, skip visualization
		return


document.add_root(layout)
document.add_periodic_callback(update_data, 5000)
update_data()
