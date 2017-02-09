from bokeh.embed import components
from bokeh.models import Title, Range1d
from bokeh.plotting import figure
from bokeh.resources import INLINE
from bokeh.layouts import row

from visualizer.modules.helpers import get_content, get_pickle


def training_progress(filename, file_path):

	results_path = file_path.replace(filename, 'results')

	accuracy = "batch_accuracy.txt"
	loss = "batch_loss.txt"

	accuracy_data = get_content(accuracy, results_path)
	loss_data = get_content(loss, results_path)

	if not accuracy_data or not loss_data:
		return None

	accuracy_plot = create_figure(accuracy_data, accuracy)
	loss_plot = create_figure(loss_data, loss)

	js_resources = INLINE.render_js()
	css_resources = INLINE.render_css()

	script, div = components(row(accuracy_plot, loss_plot))

	return js_resources, css_resources, script, div


def create_figure(content, name):

	# create new plot
	fig = figure(toolbar_location=None, plot_width=300, plot_height=300)
	fig.line(list(range(0, len(content))), content)

	# set plot title and labels
	x_label, y_label = name.split('.')[0].split('_')
	fig.title.text = ('%s Over %s' % (y_label, x_label)).title()

	fig.add_layout(Title(text=x_label.title()), "below")
	fig.add_layout(Title(text=y_label.title()), "left")

	# set limits of x to be outermost points
	fig.x_range = Range1d(0, len(content) - 1)

	return fig


# WIP: visualize activations
'''def activations(filename, file_path):

	results_path = file_path.replace(filename, 'results')
	pickle_file = "layer_dict_0.pickle"

	activations_data = get_pickle(pickle_file, results_path)'''

