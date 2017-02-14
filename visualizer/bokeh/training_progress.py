from os.path import join
from bokeh.models import ColumnDataSource, Line, Div, Paragraph
from bokeh.plotting import figure, curdoc
from bokeh.layouts import row, layout

document = curdoc()

accuracy_fig = figure(toolbar_location=None, plot_width=300, plot_height=300)
accuracy_fig.title.text = ('Accuracy over Batch')
accuracy_fig.xaxis.axis_label = 'Batch'
accuracy_fig.yaxis.axis_label = 'Accuracy'

loss_fig = figure(toolbar_location=None, plot_width=300, plot_height=300)
loss_fig.title.text = ('Loss over Batch')
loss_fig.xaxis.axis_label = 'Batch'
loss_fig.yaxis.axis_label = 'Loss'

file_source = ColumnDataSource(data=dict(file_path=[], file=[]), name='file_source')
accuracy_source = ColumnDataSource(data=dict(x=[], y=[]))
loss_source = ColumnDataSource(data=dict(x=[], y=[]))

accuracy_fig.add_glyph(accuracy_source, Line(x='x', y='y', line_color='blue'))
loss_fig.add_glyph(loss_source, Line(x='x', y='y', line_color='blue'))


def update_data():
	file_path = document.get_model_by_name('file_source').data['file_path'][0]
	filename = document.get_model_by_name('file_source').data['file'][0]
	results_path = file_path.replace(filename, 'results')

	try:
		with open(join(results_path, 'batch_accuracy.txt'), 'r') as f:
			accuracy_data = [float(line) for line in f]
		with open(join(results_path, 'batch_loss.txt'), 'r') as f:
			loss_data = [float(line) for line in f]
	except FileNotFoundError:
		accuracy_data = []
		loss_data = []

	new_accuracy_data = dict(
		x=range(0, len(accuracy_data)),
		y=accuracy_data
	)

	new_loss_data = dict(
		x=range(0, len(loss_data)),
		y=loss_data
	)

	accuracy_source.data = new_accuracy_data
	loss_source.data = new_loss_data


div = Div(text="<h3>Visualization of the training progress</h3>", width=500)

l = layout([
	[div],
	[accuracy_fig, loss_fig]
])

document.add_root(l)
document.add_root(file_source)
document.add_periodic_callback(update_data, 200)
