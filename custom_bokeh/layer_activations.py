from os.path import join
from bokeh.io import curdoc
from bokeh.layouts import gridplot
from bokeh.models import Div, ColumnDataSource, Paragraph, Column, BoxZoomTool, ResetTool
from bokeh.plotting import figure
import pickle
import numpy as np
import math

from visualizer.config import UPLOAD_FOLDER

document = curdoc()

args = document.session_context.request.arguments

#TODO: throw error if these are not provided
file = args['file'][0].decode('ascii')
user = args['user'][0].decode('ascii')

# find path for result data
results_path = join(UPLOAD_FOLDER, user, file, 'results')

# used to determine if the data source should be created
# this is needed to avoid error when the script is just started
create_source = True

div = Div(text="<h3>Visualization of the layer activations</h3>", width=500)
layout = Column(children=[div])

p = Paragraph(text="There seems to be no visualizations produced yet.", width=500)
layout.children.append(p)

layer_activation_source = ColumnDataSource(data=dict())


def fill_data_source(layer_activation_data):

	p.text = "Visualizations are being produced..."
	figures = []

	for layer_name, filters in layer_activation_data:

		# The filters are either images or just a long array of numbers
		if len(filters.shape) == 3:

			# line the filters up horizontally and pad them with white to separate the filters
			images = np.hstack([np.pad(f[::-1], 1, 'constant', constant_values=255) for f in filters])

			total_image_width = images.shape[1]

			# TODO check if this always looks good, may need to generalize
			# if there are more than 16 filters, split into 8 rows.
			no_of_rows = 8
			no_of_cols = 16
			if len(filters) > no_of_cols:
				step = math.ceil(total_image_width / no_of_rows)
				images = np.vstack([images[:, x:x + step] for x in range(0, total_image_width, step)])

			total_image_height = images.shape[0]
			total_image_width = images.shape[1]

			# add image to the data source
			layer_activation_source.add([images], name=layer_name)
			# create the figure for the current layer
			fig = create_figure(layer_activation_source, layer_name, total_image_width, total_image_height, total_image_width, total_image_height)
			figures.append(fig)

		elif len(filters.shape) == 1:

			width = filters.shape[0]

			# add image to the data source, we need to add an extra axis to plot the 1d sequence as an image
			layer_activation_source.add([filters[np.newaxis, :]], name=layer_name)
			# create the figure for the current layer
			fig = create_figure(layer_activation_source, layer_name, width, 50, width, 1)
			figures.append(fig)

	grid = gridplot(figures, ncols=1, merge_tools=False)
	layout.children.append(grid)
	p.text = ""


def create_figure(source, image_name, plot_width, plot_height, dw, dh):
	fig = figure(tools="save", plot_width=plot_width, plot_height=plot_height, x_range=(0, dw), y_range=(0, dh))
	fig.title.text = image_name
	fig.image(image=image_name, x=0, y=0, dw=dw, dh=dh, source=source)
	fig.outline_line_color = "black"
	fig.outline_line_width = 3
	fig.axis.visible = False
	fig.toolbar.logo = None
	fig.toolbar_location = "left"
	fig.toolbar_sticky = False
	if plot_height > 100:
		fig.add_tools(BoxZoomTool(match_aspect=True))
		fig.add_tools(ResetTool())
	return fig


def update_data():
	global create_source
	try:
		with open(join(results_path, 'layer_activations.pickle'), 'rb') as f:
			layer_activation_data = pickle.load(f)

		# if it is the first time data is detected, we need to fill the data source with the layers of the network
		if create_source:
			# temporary remove callback to make sure the function is not being called while creating the visualizations
			document.remove_periodic_callback(update_data)
			fill_data_source(layer_activation_data)
			create_source = False
			# allow some time for the layer activations to load TODO: generalize this
			document.add_periodic_callback(update_data, 90000)
		# if not, we can simply update the data
		else:
			# dictionary to hold new images
			new_layer_activation_data = {}

			for layer_name, filters in layer_activation_data:

				# The filters are either images or just a long array of numbers
				if len(filters.shape) == 3:

					# line the filters up horizontally and pad them with white to separate the filters
					images = np.hstack([np.pad(f[::-1], 1, 'constant', constant_values=255) for f in filters])

					total_image_width = images.shape[1]

					# TODO check if this always looks good, may need to generalize
					# if there are more than 16 filters, split into 8 rows.
					no_of_rows = 8
					no_of_cols = 16
					if len(filters) > no_of_cols:
						step = math.ceil(total_image_width / no_of_rows)
						images = np.vstack([images[:, x:x + step] for x in range(0, total_image_width, step)])

					new_layer_activation_data[layer_name] = [images]

				elif len(filters.shape) == 1:
					# need to add an extra axis to plot 1d sequence as an image
					new_layer_activation_data[layer_name] = [filters[np.newaxis, :]]

			# update all images
			layer_activation_source.data = new_layer_activation_data

	except FileNotFoundError:
		# this means layer activation data has not been created yet, skip visualization
		pass

document.add_periodic_callback(update_data, 10000)
update_data()
document.add_root(layout)

