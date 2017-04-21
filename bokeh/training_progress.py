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

accuracy_fig = figure(tools="box_zoom, reset, save", plot_width=900, plot_height=300)
accuracy_fig.title.text = ('Accuracy')
accuracy_fig.xaxis.axis_label = 'Epoch'
accuracy_fig.yaxis.axis_label = 'Accuracy'
accuracy_fig.x_range = DataRange1d(start=0)
accuracy_fig.y_range = Range1d(0, 1)
accuracy_fig.toolbar.logo = None
grid.append([accuracy_fig])

loss_fig = figure(tools="box_zoom, reset, save", plot_width=900, plot_height=300)
loss_fig.title.text = ('Loss')
loss_fig.xaxis.axis_label = 'Epoch'
loss_fig.yaxis.axis_label = 'Loss'
loss_fig.x_range = DataRange1d(start=0)
loss_fig.y_range = DataRange1d(start=0)
loss_fig.toolbar.logo = None
grid.append([loss_fig])

train_source = ColumnDataSource(data=dict(x=[], acc_y=[], loss_y=[]))
val_source = ColumnDataSource(data=dict(x=[], acc_y=[], loss_y=[]))

accuracy_fig.add_glyph(train_source, Line(x='x', y='acc_y', line_color='blue'))
accuracy_fig.add_glyph(val_source, Line(x='x', y='acc_y', line_color='green'))
loss_fig.add_glyph(train_source, Line(x='x', y='loss_y', line_color='blue'))
loss_fig.add_glyph(val_source, Line(x='x', y='loss_y', line_color='green'))


def update_data():

	data_length = len(train_source.data['x'])

	try:
		with open(join(results_path, 'training_progress.txt')) as f:
			train_data = list(zip(*[line.strip().split() for line in f]))
		if not train_data:
			return  # this means that the file has been created but no data has been added yet
		new_data_length = len(train_data[0])
		p.text = ""
	except FileNotFoundError:
		return

	# find new data that should be added to the graph
	new_train_data = dict(
		x=train_data[0][data_length:new_data_length],
		acc_y=train_data[1][data_length:new_data_length],
		loss_y=train_data[2][data_length:new_data_length]
	)

	train_source.stream(new_train_data)


document.add_root(layout(grid))
document.add_periodic_callback(update_data, 200)
