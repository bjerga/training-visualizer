from bokeh.models import Range1d, ColumnDataSource
from bokeh.plotting import figure, curdoc
from bokeh.layouts import row

from os.path import join

filename = "mnist_keras.py"
file_path = "/Users/annieaa/Documents/NTNU/Fordypningsprosjekt/visualizer/visualizer/static/user_storage/anniea/programs/mnist_keras/mnist_keras.py"

results_path = file_path.replace(filename, 'results')

accuracy = "batch_accuracy.txt"
loss = "batch_loss.txt"

try:
	with open(join(results_path, accuracy), 'r') as f:
		accuracy_data = [float(line) for line in f]
	with open(join(results_path, loss), 'r') as f:
		loss_data = [float(line) for line in f]
except FileNotFoundError:
	accuracy_data = None
	loss_data = None

if accuracy_data and loss_data:

	# create new plot for accuracy
	fig1 = figure(toolbar_location=None, plot_width=300, plot_height=300)
	fig1.line(list(range(0, len(accuracy_data))), accuracy_data)

	# set plot title and labels
	x_label, y_label = accuracy.split('.')[0].split('_')
	fig1.title.text = ('%s Over %s' % (y_label, x_label)).title()

	fig1.xaxis.axis_label = x_label.title()
	fig1.yaxis.axis_label = y_label.title()

	# set limits of x to be outermost points
	fig1.x_range = Range1d(0, len(accuracy_data) - 1)

	# create new plot for loss
	fig2 = figure(toolbar_location=None, plot_width=300, plot_height=300)
	fig2.line(list(range(0, len(loss_data))), loss_data)

	# set plot title and labels
	x_label, y_label = loss.split('.')[0].split('_')
	fig2.title.text = ('%s Over %s' % (y_label, x_label)).title()

	fig2.xaxis.axis_label = x_label.title()
	fig2.yaxis.axis_label = y_label.title()

	# set limits of x to be outermost points
	fig2.x_range = Range1d(0, len(loss_data) - 1)

	curdoc().add_root(row(fig1, fig2))
