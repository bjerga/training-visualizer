from os.path import join
from bokeh.models import ColumnDataSource, Line, Div, Paragraph, SingleIntervalTicker, Range1d, DataRange1d, Circle, \
	HoverTool
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

p = Paragraph(text="There seems to be no training progress data produced yet.", width=600)
grid.append([p])


def create_figure(title, label_x, label_y, x_range, y_range, tools="box_zoom, reset, save"):
	fig = figure(tools=tools, plot_width=900, plot_height=300)
	fig.title.text = title
	fig.xaxis.axis_label = label_x
	fig.yaxis.axis_label = label_y
	fig.x_range = x_range
	fig.y_range = y_range
	fig.toolbar.logo = None
	return fig


def create_circle(y, size=6):
	return Circle(x='x', y=y, line_color='green', fill_color='green', fill_alpha=0.5, size=size)


def create_hover_tool(render, y_title, y_value):
	return HoverTool(renderers=[render], tooltips=[('Epoch', '@x'), (y_title, y_value)])


# create the accuracy figure
accuracy_fig = create_figure('Accuracy', 'Epoch', 'Accuracy', DataRange1d(start=0), Range1d(0, 1))
grid.append([accuracy_fig])

# create the loss figure
loss_fig = create_figure('Loss', 'Epoch', 'Loss', DataRange1d(start=0), DataRange1d(start=0))
grid.append([loss_fig])

# Initialize sources for holding the data
train_source = ColumnDataSource(data=dict(x=[], acc_y=[], loss_y=[]))
val_source = ColumnDataSource(data=dict(x=[], acc_y=[], loss_y=[]))

# add lines for training accuracy and circles for validation accuracy
accuracy_fig.add_glyph(train_source, Line(x='x', y='acc_y', line_color='blue'))
acc_render = accuracy_fig.add_glyph(val_source, create_circle('acc_y'))

# add a hover tool to validation accuracy circles
accuracy_fig.add_tools(create_hover_tool(acc_render, 'Validation accuracy', '@acc_y'))

# add lines for training loss and circles for validation loss
loss_fig.add_glyph(train_source, Line(x='x', y='loss_y', line_color='blue'))
loss_render = loss_fig.add_glyph(val_source, create_circle('loss_y'))

# add a hover tool to validation loss circles
loss_fig.add_tools(create_hover_tool(loss_render, 'Validation loss', '@loss_y'))


def update_data():

	data_length = len(train_source.data['x'])
	val_data_length = len(val_source.data['x'])

	try:
		with open(join(results_path, 'training_progress.txt')) as f:
			training_progress_data = list(zip(*[line.strip().split() for line in f]))
		if not training_progress_data:
			# this means that the file has been created but no data has been added yet, skip visualization
			return
		new_data_length = len(training_progress_data[0])
		p.text = ""
	except FileNotFoundError:
		# this means that file has not been created yet, skip visualization
		return

	try:
		with open(join(results_path, 'training_progress_val.txt')) as f:
			validation_progress_data = list(zip(*[line.strip().split() for line in f]))
		new_val_data_length = len(validation_progress_data)
	except FileNotFoundError:
		# this means that no validation data has been created, set to empty
		validation_progress_data = []
		new_val_data_length = 0

	# find new data that should be added to the graph and stream it
	new_training_progress_data = dict(
		x=training_progress_data[0][data_length:new_data_length],
		acc_y=training_progress_data[1][data_length:new_data_length],
		loss_y=training_progress_data[2][data_length:new_data_length]
	)
	train_source.stream(new_training_progress_data)

	# if there is any data for validation, find the new data that should be added to the graph and stream it
	if validation_progress_data:
		new_validation_progress_data = dict(
			x=validation_progress_data[0][val_data_length:new_val_data_length],
			acc_y=validation_progress_data[1][val_data_length:new_val_data_length],
			loss_y=validation_progress_data[2][val_data_length:new_val_data_length]
		)
		val_source.stream(new_validation_progress_data)


document.add_root(layout(grid))
document.add_periodic_callback(update_data, 200)
