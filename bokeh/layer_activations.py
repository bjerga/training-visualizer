from os.path import join
from bokeh.io import curdoc
from bokeh.layouts import column
from bokeh.models import Div, ColumnDataSource, Paragraph, Column
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


def create_figures(layer_activation_data):

	p.text = "Visualizations are being produced..."
	figures = []

	for layer_name, filters in layer_activation_data:

		# The filters are either images or just a long array of numbers
		if len(filters.shape) == 3:
			# line the filters up horizontally
			images = np.hstack([f[::-1] for f in filters])

			total_image_width = images.shape[1]

			# if there are more than 4 filters, split into several rows.
			if len(filters) > 4:
				step = math.ceil(total_image_width / 4)
				images = np.vstack([images[:, x:x + step] for x in range(0, total_image_width, step)])

			total_image_height = images.shape[0]
			total_image_width = images.shape[1]

			layer_activation_source.add([images], name=layer_name)

			fig = figure(tools="box_zoom, reset, save", plot_width=total_image_width*5, plot_height=total_image_height*5, x_range=(0, total_image_width), y_range=(0, total_image_height))
			fig.title.text = layer_name
			fig.image(image=layer_name, x=0, y=0, dw=total_image_width, dh=total_image_height, source=layer_activation_source)
			fig.axis.visible = False
			fig.toolbar.logo = None

			figures.append(fig)

		elif len(filters.shape) == 1:

			width = filters.shape[0]

			# need to add an axis to plot as image
			layer_activation_source.add([filters[np.newaxis, :]], name=layer_name)

			fig = figure(tools="save", plot_width=width*10, plot_height=50, x_range=(0, width), y_range=(0, 1))
			fig.title.text = layer_name
			fig.image(image=layer_name, x=0, y=0, dw=width, dh=1, source=layer_activation_source)
			fig.axis.visible = False
			fig.toolbar.logo = None

			figures.append(fig)

	layout.children.append(column(figures))
	p.text = ""


def update_data():
	global create_source
	try:
		with open(join(results_path, 'layer_activations.pickle'), 'rb') as f:
			layer_activation_data = pickle.load(f)

		# if it is the first time data is detected, we need to initialize the data source
		if create_source:
			# temporary remove callback to make sure the function is not being called while creating the visualizations
			document.remove_periodic_callback(update_data)
			create_figures(layer_activation_data)
			create_source = False
			document.add_periodic_callback(update_data, 5000)
		# if not, we can simply update the data
		else:
			for layer_name, filters in layer_activation_data:

				if len(filters.shape) == 3:
					images = np.hstack([f[::-1] for f in filters])

					total_image_width = images.shape[1]

					if len(filters) > 4:
						step = math.ceil(total_image_width / 4)
						images = np.vstack([images[:, x:x + step] for x in range(0, total_image_width, step)])

					layer_activation_source.data[layer_name] = [images]

				elif len(filters.shape) == 1:

					layer_activation_source.data[layer_name] = [filters[np.newaxis, :]]

	except FileNotFoundError:
		pass

document.add_periodic_callback(update_data, 5000)
update_data()
document.add_root(layout)

