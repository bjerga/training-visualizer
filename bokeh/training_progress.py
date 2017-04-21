from os.path import join
from bokeh.models import ColumnDataSource, Line, Div, Paragraph, SingleIntervalTicker, Range1d, DataRange1d
from bokeh.plotting import figure, curdoc
from bokeh.layouts import layout

from visualizer.config import UPLOAD_FOLDER

document = curdoc()

args = document.session_context.request.arguments

#TODO: throw error if these are not provided
file = args['file'][0].decode('ascii')
user = args['user'][0].decode('ascii')

results_path = join(UPLOAD_FOLDER, user, file, 'results')

grid = []

div = Div(text="<h3>Visualization of the training progress</h3>", width=500)
grid.append([div])

p = Paragraph(text="There seems to be no training progress data produced yet.", width=500)
grid.append([p])

x_ticker = SingleIntervalTicker(interval=1, num_minor_ticks=10)

accuracy_fig = figure(tools="box_zoom, reset, save", plot_width=900, plot_height=300)
accuracy_fig.title.text = ('Accuracy')
accuracy_fig.xaxis.axis_label = 'Epoch'
accuracy_fig.yaxis.axis_label = 'Accuracy'
accuracy_fig.xaxis.ticker = x_ticker
accuracy_fig.x_range = DataRange1d(start=0)
accuracy_fig.y_range = Range1d(0, 1)
accuracy_fig.toolbar.logo = None
grid.append([accuracy_fig])

loss_fig = figure(tools="box_zoom, reset, save", plot_width=900, plot_height=300)
loss_fig.title.text = ('Loss')
loss_fig.xaxis.axis_label = 'Epoch'
loss_fig.yaxis.axis_label = 'Loss'
loss_fig.xaxis.ticker = x_ticker
loss_fig.x_range = DataRange1d(start=0)
loss_fig.y_range = DataRange1d(start=0)
loss_fig.toolbar.logo = None
grid.append([loss_fig])

train_source = ColumnDataSource(data=dict(acc_x=[], acc_y=[], loss_x=[], loss_y=[]))
val_source = ColumnDataSource(data=dict(acc_x=[], acc_y=[], loss_x=[], loss_y=[]))

accuracy_fig.add_glyph(train_source, Line(x='acc_x', y='acc_y', line_color='blue'))
accuracy_fig.add_glyph(val_source, Line(x='acc_x', y='acc_y', line_color='green'))
loss_fig.add_glyph(train_source, Line(x='loss_x', y='loss_y', line_color='blue'))
loss_fig.add_glyph(val_source, Line(x='loss_x', y='loss_y', line_color='green'))


def update_data():

	try:
		with open(join(results_path, 'accuracy_train.txt'), 'r') as f:
			accuracy_train_data = list(zip(*[line.strip().split() for line in f]))
		with open(join(results_path, 'loss_train.txt'), 'r') as f:
			loss_train_data = list(zip(*[line.strip().split() for line in f]))
		p.text = ""
	except FileNotFoundError:
		return

	new_train_data = dict(
		acc_x=accuracy_train_data[0],
		acc_y=accuracy_train_data[1],
		loss_x=loss_train_data[0],
		loss_y=loss_train_data[1]
	)
	train_source.data = new_train_data


document.add_root(layout(grid))
document.add_periodic_callback(update_data, 1000)
