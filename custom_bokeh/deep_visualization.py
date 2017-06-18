from bokeh.io import curdoc
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource, Paragraph, Column, BoxZoomTool, Range1d

from os.path import join

from bokeh.plotting import figure
import pickle

from visualizer.config import UPLOAD_FOLDER
from custom_bokeh.helpers import *

document = curdoc()

args = document.session_context.request.arguments

try:
	file = args['file'][0].decode('ascii')
	user = args['user'][0].decode('ascii')
except KeyError as e:
	raise KeyError(str(e) + '. Filename and username must be provided as request parameters.')

# find path for result data
results_path = join(UPLOAD_FOLDER, user, file, 'results')

deep_visualization_source = ColumnDataSource(data=dict())

p = Paragraph(text="", width=500)
layout = Column(children=[p])


def fill_data_source(deep_visualization_data):

	p.text = "Visualizations are being produced..."
	figures = []

	# loop through the units
	for array, layer_name, unit_index, loss_value in deep_visualization_data:

		name = "{}_{}".format(layer_name, unit_index)
		title = "Unit at index {} in {}".format(unit_index, layer_name)

		img_width = array.shape[1]
		img_height = array.shape[0]

		fig = figure(title=title, tools="reset, save, pan", plot_width=250, plot_height=250,
						x_range=Range1d(0, img_width, bounds=(0, img_width)),
						y_range=Range1d(0, img_height, bounds=(0, img_height)),
						outline_line_color="black", outline_line_width=3)
		fig.add_tools(BoxZoomTool(match_aspect=True))
		fig.axis.visible = False

		add_image_from_source(fig, deep_visualization_source, array, name)

		figures.append(fig)

	# make a grid of the units
	grid = gridplot(figures, ncols=4, toolbar_options=dict(logo=None))
	layout.children.append(grid)
	p.text = ""


def update_data():
	try:
		with open(join(results_path, 'deep_visualization.pickle'), 'rb') as f:
			deep_visualization_data = pickle.load(f)

		# if it is the first time data is detected, we need to fill the data source with the images
		if not deep_visualization_source.data:
			# temporary remove callback to make sure the function is not being called while creating the visualizations
			document.remove_periodic_callback(update_data)
			fill_data_source(deep_visualization_data)
			document.add_periodic_callback(update_data, 5000)
		# if not, we can simply update the data
		else:
			for array, layer_name, unit_index, loss_value in deep_visualization_data:
				name = "{}_{}".format(layer_name, unit_index)
				img = process_image_dim(array)
				deep_visualization_source.data[name] = [img[::-1]]

	except FileNotFoundError:
		# this means deconvolution data has not been created yet, skip visualization
		p.text = "There are no visualization data produced yet."
		return
	except EOFError:
		# this means deconvolution data has been created, but is empty, skip visualization
		p.text = "There are no visualization data produced yet."
		return

document.add_root(layout)
document.add_periodic_callback(update_data, 5000)
update_data()
