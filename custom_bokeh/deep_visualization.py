from bokeh.io import curdoc
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource, Div, Paragraph, Column, FixedTicker

from os.path import join

from bokeh.plotting import figure
import pickle

from visualizer.config import UPLOAD_FOLDER

document = curdoc()

args = document.session_context.request.arguments

# TODO: throw error if these are not provided
file = args['file'][0].decode('ascii')
user = args['user'][0].decode('ascii')

# find path for result data
results_path = join(UPLOAD_FOLDER, user, file, 'results')

# used to determine if the data source should be created
# this is needed to avoid error when the script is just started
create_source = True

div = Div(text="<h3>Deep Visualizations</h3>", width=500)
layout = Column(children=[div])

p = Paragraph(text="There seems to be no visualizations produced yet.", width=500)
layout.children.append(p)

deep_visualization_source = ColumnDataSource(data=dict())


def fill_data_source(deep_visualization_data):

	p.text = "Visualizations are being produced..."
	figures = []

	for array, layer_name, neuron_no, loss_value in deep_visualization_data:

		name = "{}_{}".format(layer_name, neuron_no)
		title = "Neuron #{} in {}".format(neuron_no, layer_name)

		# add image to the data source
		deep_visualization_source.add([array[::-1]], name=name)

		fig = create_figure(deep_visualization_source, name, title, array.shape[0], array.shape[1])
		figures.append(fig)

	document.add_root(gridplot(figures, ncols=2, toolbar_options=dict(logo=None)))
	p.text = ""


def create_figure(source, image_name, title, dw, dh, tools="box_zoom, reset, save"):
	fig = figure(tools=tools, plot_width=250, plot_height=250, x_range=(0, dw), y_range=(0, dh))
	fig.title.text = title
	fig.image(image=image_name, x=0, y=0, dw=dw, dh=dh, source=source)
	fig.outline_line_color = "black"
	fig.outline_line_width = 3
	fig.axis.visible = False
	return fig


def update_data():
	global create_source
	try:
		with open(join(results_path, 'deep_visualization.pickle'), 'rb') as f:
			deep_visualization_data = pickle.load(f)

		if create_source:
			# temporary remove callback to make sure the function is not being called while creating the visualizations
			document.remove_periodic_callback(update_data)
			fill_data_source(deep_visualization_data)
			create_source = False
			document.add_periodic_callback(update_data, 5000)
		else:
			for array, layer_name, neuron_no, loss_value in deep_visualization_data:
				name = "{}_{}".format(layer_name, neuron_no)
				deep_visualization_source.data[name] = [array[::-1]]

	except FileNotFoundError:
		return
	except EOFError:
		return

document.add_root(layout)
document.add_periodic_callback(update_data, 5000)
update_data()
