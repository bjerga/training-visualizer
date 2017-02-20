from os.path import join
from bokeh.io import curdoc
from bokeh.layouts import column, layout
from bokeh.models import Div, ColumnDataSource
from bokeh.plotting import figure
import pickle
import numpy as np
import math

document = curdoc()

file_source = ColumnDataSource(data=dict(file_path=[], file=[]), name='file_source')
layer_activation_source = ColumnDataSource(data=dict())

results_path = "/Users/annieaa/Documents/NTNU/Fordypningsprosjekt/visualizer/visualizer/static/user_storage/anniea/" \
			   "programs/mnist_keras/results"

# read content of pickle file
with open(join(results_path, 'layer_dict.pickle'), 'rb') as f:
	layer_activation_data = pickle.load(f)


grid = []

div = Div(text="<h3>Visualization of the layer activations</h3>", width=500)

grid.append([div])

#for layer_no in range(len(layer_activation_data)):
for layer_no in range(0, 5):

	layer_name, layer_activation = layer_activation_data[layer_no]

	print(layer_activation.shape)

	# scale to fit between [0.0, 255.0]
	layer_activation += max(-np.min(layer_activation), 0.0)
	la_max = np.max(layer_activation)
	if la_max != 0.0:
		layer_activation /= la_max
		layer_activation *= 255.0

	filters = np.transpose(layer_activation[0], (2, 0, 1))

	no_of_filters = filters.shape[0]

	if no_of_filters == 1:

		plot_width = 250
		plot_height = 250

		x_range_end = 20
		y_range_end = 20

		total_image = filters[0]

	else:

		plot_width = 500
		plot_height = 250

		x_range_end = 40
		y_range_end = 20

		# needs to be generalized
		no_of_cols = 8
		no_of_rows = math.ceil(no_of_filters/no_of_cols)

		rows = []

		for i in range(no_of_rows):
			rows.append(np.hstack(tuple(np.lib.pad(filters[j], (1, 1), 'constant', constant_values=255) for j in range(i, i+8))))

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



'''grid = []

div = Div(text="<h3>Visualization of the layer activations</h3>", width=500)

grid.append([div])

for layer_no in range(0, 5):
	layer_name, _ = layer_activation_data[layer_no]

	title = "Layer " + str(layer_no) + ": " + layer_name

	fig ='''
	


# for all layers
#for layer_no in range(len(layer_activation_data)):
'''for layer_no in range(0, 5):

	layer_name, layer_activation = layer_activation_data[layer_no]

	title = "Layer " + str(layer_no) + ": " + layer_name

	# scale to fit between [0.0, 255.0]
	layer_activation += max(-np.min(layer_activation), 0.0)
	la_max = np.max(layer_activation)
	if la_max != 0.0:
		layer_activation /= la_max
		layer_activation *= 255.0

	fig = figure(title=title, tools="box_zoom, reset, save", x_range=(0, 40), y_range=(0, 20), plot_width=500, plot_height=250)
	fig.axis.visible = False
	fig.logo = None

	filters = np.transpose(layer_activation[0], (2, 1, 0))
	no_of_filters = layer_activation.shape[3]

	if no_of_filters > 1:
		rows = 4
		cols = 8
	else:
		rows = 1
		cols = no_of_filters

	x = 0
	y = 5 * (rows - 1)

	for filter_no in range(no_of_filters):
		fig.image(image=[filters[filter_no].astype('uint8')], x=x, y=y, dw=5, dh=5)
		x += 5
		if x % 40 == 0:
			x = 0
			y -= 5

	grid.append([fig])


document.add_root(layout(grid))'''
