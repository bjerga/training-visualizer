from os.path import join
from bokeh.io import curdoc
from bokeh.layouts import column, layout
from bokeh.models import Div, ColumnDataSource
from bokeh.plotting import figure
import pickle
import numpy as np

document = curdoc()

# file_source = ColumnDataSource(data=dict(file_path=[], file=[]), name='file_source')
# layer_activation_source = ColumnDataSource(data=dict(x=[], y=[]))


results_path = "/Users/annieaa/Documents/NTNU/Fordypningsprosjekt/visualizer/visualizer/static/user_storage/anniea/" \
			   "programs/mnist_keras/results"

# read content of pickle file
with open(join(results_path, 'layer_dict_7.pickle'), 'rb') as f:
	layer_activation_data = pickle.load(f)

grid = []

div = Div(text="<h3>Visualization of the layer activations</h3>", width=500)

grid.append([div])

# for all layers
#for layer_no in range(len(layer_activation_data)):
for layer_no in range(0, 5):

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


document.add_root(layout(grid))
