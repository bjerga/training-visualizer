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

accuracy_fig = create_figure('Accuracy', 'Epoch', 'Accuracy', DataRange1d(start=0), Range1d(0, 1))

grid.append([accuracy_fig])

loss_fig = create_figure('Loss', 'Epoch', 'Loss', DataRange1d(start=0), DataRange1d(start=0))
grid.append([loss_fig])

train_source = ColumnDataSource(data=dict(x=[], acc_y=[], loss_y=[]))
val_source = ColumnDataSource(data=dict(x=[], acc_y=[], loss_y=[]))

accuracy_fig.add_glyph(train_source, Line(x='x', y='acc_y', line_color='blue'))
acc_render = accuracy_fig.add_glyph(val_source, Circle(x='x', y='acc_y', line_color='green', fill_color='green',
													   fill_alpha=0.5, size=6))

acc_hover = HoverTool(renderers=[acc_render], tooltips=[('Epoch', '@x'), ('Validation accuracy', '@acc_y')])
accuracy_fig.add_tools(acc_hover)

loss_fig.add_glyph(train_source, Line(x='x', y='loss_y', line_color='blue'))
loss_render = loss_fig.add_glyph(val_source, Circle(x='x', y='loss_y', line_color='green', fill_color='green',
													fill_alpha=0.5, size=6))

loss_hover = HoverTool(renderers=[loss_render], tooltips=[('Epoch', '@x'), ('Validation loss', '@loss_y')])
loss_fig.add_tools(loss_hover)


def update_data():

	plot_val = True

	data_length = len(train_source.data['x'])
	val_data_length = len(val_source.data['x'])

	try:
		with open(join(results_path, 'training_progress.txt')) as f:
			train_data = list(zip(*[line.strip().split() for line in f]))
		if not train_data:
			return  # this means that the file has been created but no data has been added yet
		new_data_length = len(train_data[0])
		p.text = ""
	except FileNotFoundError:
		return

	try:
		with open(join(results_path, 'training_progress_val.txt')) as f:
			val_data = list(zip(*[line.strip().split() for line in f]))
		new_val_data_length = len(val_data)
		if not val_data:
			plot_val = False
	except FileNotFoundError:
		plot_val = False

	# find new data that should be added to the graph
	new_train_data = dict(
		x=train_data[0][data_length:new_data_length],
		acc_y=train_data[1][data_length:new_data_length],
		loss_y=train_data[2][data_length:new_data_length]
	)

	train_source.stream(new_train_data)

	if plot_val:
		new_val_data = dict(
			x=val_data[0][val_data_length:new_val_data_length],
			acc_y=val_data[1][val_data_length:new_val_data_length],
			loss_y=val_data[2][val_data_length:new_val_data_length]
		)

		val_source.stream(new_val_data)


document.add_root(layout(grid))
document.add_periodic_callback(update_data, 200)
