from os import listdir
from os.path import join
from bokeh.io import curdoc
from bokeh.layouts import gridplot
from bokeh.models import Div, ColumnDataSource, Paragraph, Column, BoxZoomTool, ResetTool, Range1d, PanTool, SaveTool
from bokeh.plotting import figure
from PIL import Image
import pickle
import math

from visualizer.config import UPLOAD_FOLDER
from custom_bokeh.utils import *

document = curdoc()

args = document.session_context.request.arguments

try:
	file = args['file'][0].decode('ascii')
	user = args['user'][0].decode('ascii')
except KeyError as e:
	raise KeyError(str(e) + '. Filename and username must be provided as request parameters.')

# find path for result data
results_path = join(UPLOAD_FOLDER, user, file, 'results')

p = Paragraph(text="", width=500)
layout = Column(children=[p])

layer_activation_source = ColumnDataSource(data=dict())


def create_image_grid(filters):

	no_of_images = len(filters)

	if no_of_images < 4:
		no_of_rows = 1
	elif no_of_images < 64:
		no_of_rows = 4
	else:
		no_of_rows = 8

	no_of_cols = math.ceil(no_of_images/no_of_rows)

	# line the filters up horizontally and pad them with white to separate the filters
	images = np.hstack([np.pad(f, 1, 'constant', constant_values=255) for f in filters])

	total_width = images.shape[1]

	if no_of_images > no_of_cols:
		step = math.ceil(total_width / no_of_rows)
		images = np.vstack([images[:, x:x + step] for x in range(0, total_width, step)])

	return images


def fill_data_source(layer_activation_data):

	p.text = "Visualizations are being produced..."

	# get original image
	try:
		images_folder = join(UPLOAD_FOLDER, user, file, 'images')
		image_name = listdir(images_folder)[-1]
		orig_img = np.array(Image.open(join(images_folder, image_name)))
	except FileNotFoundError:
		raise FileNotFoundError(str(e) + '. No visualization input image found in the image folder')

	orig_img_height = orig_img.shape[0]
	orig_img_width = orig_img.shape[1]

	if orig_img_width < 100:
		img_scale = math.ceil(100 / orig_img_width)
	elif orig_img_height < 100:
		img_scale = math.ceil(100 / orig_img_height)
	else:
		img_scale = 1

	# create plot for the original image
	orig_img_fig = figure(title="Input", plot_width=orig_img_width * img_scale, plot_height=orig_img_height * img_scale,
						  x_range=Range1d(0, orig_img_width, bounds=(0, orig_img_width)),
						  y_range=Range1d(0, orig_img_height, bounds=(0, orig_img_height)),
						  tools="pan, reset", toolbar_location="left", toolbar_sticky=False,
						  outline_line_color="black", outline_line_width=3)
	orig_img_fig.add_tools(BoxZoomTool(match_aspect=True))
	orig_img_fig.axis.visible = False
	orig_img_fig.toolbar.logo = None

	add_image_from_array(orig_img_fig, orig_img)

	figures = [orig_img_fig]

	for layer_name, filters in layer_activation_data:

		# The filters are either images or just a long array of numbers
		if len(filters.shape) == 3:

			images = create_image_grid(filters)
			height = images.shape[0]
			width = images.shape[1]

			fig = figure(title=layer_name, plot_width=width*img_scale, plot_height=height*img_scale,
							x_range=Range1d(0, width, bounds=(0, width)),
							y_range=Range1d(0, height, bounds=(0, height)),
							tools="", toolbar_location="left", toolbar_sticky=False,
							outline_line_color="black", outline_line_width=3)
			fig.axis.visible = False
			fig.toolbar.logo = None

			if height > 100:
				fig.add_tools(PanTool())
				fig.add_tools(BoxZoomTool(match_aspect=True))
				fig.add_tools(ResetTool())
			# only save tool if the resulting visualization is small
			fig.add_tools(SaveTool())

			add_image_from_source(fig, layer_activation_source, images, layer_name, always_grayscale=True)
			figures.append(fig)

		elif len(filters.shape) == 1:

			width = filters.shape[0]
			# scale up width if the original width is too small
			if width < 300:
				width_scale = math.ceil(300/width)
			else:
				width_scale = 1
			height = 1

			fig = figure(title=layer_name, tools="save", plot_width=width*width_scale, plot_height=50,
							x_range=Range1d(0, width, bounds=(0, width)),
							y_range=Range1d(0, height, bounds=(0, height)),
							toolbar_location="left", toolbar_sticky=False,
							outline_line_color="black", outline_line_width=3)
			fig.axis.visible = False
			fig.toolbar.logo = None

			add_image_from_source(fig, layer_activation_source, filters[np.newaxis, :], layer_name, always_grayscale=True)
			figures.append(fig)

	grid = gridplot(figures, ncols=1, merge_tools=False)
	layout.children.append(grid)
	p.text = ""


def update_data():
	try:
		with open(join(results_path, 'layer_activations.pickle'), 'rb') as f:
			layer_activation_data = pickle.load(f)

		# if the data source is empty, we need to fill the data source with the layers of the network
		if not layer_activation_source.data:
			# temporary remove callback to make sure the function is not being called while creating the visualizations
			document.remove_periodic_callback(update_data)
			fill_data_source(layer_activation_data)
			# allow some time for the layer activations to load TODO: generalize this
			document.add_periodic_callback(update_data, 90000)
		# if there is data in the data source, we can simply update this data
		else:
			# dictionary to hold new images
			new_layer_activation_data = {}

			for layer_name, filters in layer_activation_data:

				# The filters are either images or just a long array of numbers
				if len(filters.shape) == 3:
					# create a grid of images
					images = create_image_grid(filters)
					new_layer_activation_data[layer_name] = [images[::-1]]

				elif len(filters.shape) == 1:
					# need to add an extra axis to plot 1d sequence as an image
					new_layer_activation_data[layer_name] = [filters[np.newaxis, :]]

			# update all images
			layer_activation_source.data = new_layer_activation_data

	except FileNotFoundError:
		p.text = "There are no visualizations produced yet."
		# this means layer activation data has not been created yet, skip visualization
		pass

document.add_periodic_callback(update_data, 10000) #TODO: generalize
update_data()
document.add_root(layout)

