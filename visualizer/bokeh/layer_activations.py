from os.path import join
from bokeh.io import curdoc
from bokeh.layouts import layout
from bokeh.models import Div, ColumnDataSource
from bokeh.plotting import figure
import pickle
import numpy as np
import math

document = curdoc()

args = document.session_context.request.arguments

#TODO: throw error if these are not provided
file = args['file'][0].decode('ascii')
user = args['user'][0].decode('ascii')

layer_activation_source = ColumnDataSource(data=dict())


#TODO: get upload folder from a config file instead
results_path = "/Users/annieaa/Documents/NTNU/Fordypningsprosjekt/visualizer/visualizer/static/user_storage/" + \
			   user + "/" + file + "/results"

# read content of pickle file
try:
	with open(join(results_path, 'layer_activations.pickle'), 'rb') as f:
		layer_activation_data = pickle.load(f)
except FileNotFoundError:
	#TODO: provide a div text saying that visualization could not be retrieved
	layer_activation_data = []
	print('Cannot find file')

grid = []

div = Div(text="<h3>Visualization of the layer activations</h3>", width=500)

grid.append([div])

#TODO: decide on how to choose which layers should be displayed
#for layer_no in range(len(layer_activation_data)):
for layer_no in range(0, 5):

	layer_name, layer_activation = layer_activation_data[layer_no]

	# scale to fit between [0.0, 255.0]
	layer_activation += max(-np.min(layer_activation), 0.0)
	la_max = np.max(layer_activation)
	if la_max != 0.0:
		layer_activation /= la_max
		layer_activation *= 255.0

	filters = np.transpose(layer_activation[0], (2, 0, 1))

	no_of_filters = filters.shape[0]

	#TODO: Should be generalized more
	if no_of_filters == 1:

		plot_width, plot_height = 250, 250
		x_range_end, y_range_end = 20, 20
		total_image = np.flipud(filters[0])

	else:

		plot_width, plot_height = 500, 250
		x_range_end, y_range_end = 40, 20

		no_of_cols = 8
		no_of_rows = math.ceil(no_of_filters/no_of_cols)

		rows = []
		for i in range(no_of_rows):
			#TODO: Probably very slow, find a faster way to do this
			rows.append(np.hstack(tuple(np.flipud(np.lib.pad(filters[j], (1, 1), 'constant', constant_values=255)) for j in range(i, i+8))))

		total_image = np.vstack(tuple(rows))

	layer_activation_source.add([total_image.astype('uint8')], name=layer_name)

	title = "Layer " + str(layer_no) + ": " + layer_name
	fig = figure(title=title, tools="box_zoom, reset, save", x_range=(0, x_range_end), y_range=(0, y_range_end),
				 plot_width=plot_width, plot_height=plot_height)
	fig.image(image=layer_name, x=0, y=0, dw=x_range_end, dh=y_range_end, source=layer_activation_source)
	fig.axis.visible = False
	fig.toolbar.logo = None

	grid.append([fig])

document.add_root(layout(grid))
