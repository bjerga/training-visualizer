from bokeh.embed import components
from bokeh.plotting import figure
from bokeh.resources import INLINE


def polynomial():

	# Get all the form arguments in the url with defaults
	color = '#000000'
	_from = 0
	to = 10

	# Create a polynomial line graph with those arguments
	x = list(range(_from, to + 1))
	fig = figure(title="Polynomial", toolbar_location=None)
	fig.line(x, [i ** 2 for i in x], color=color, line_width=2)

	js_resources = INLINE.render_js()
	css_resources = INLINE.render_css()

	script, div = components(fig)
	print(div)

	return js_resources, css_resources, script, div


def plot_content(text_file, plot_color, results_path, plots_path):
	filename = text_file[:-4]



# plot text file content
def plot_content(text_file, plot_color, results_path, plots_path):

	# get filename without '.txt'
	filename = text_file[:-4]

	# read content of text file
	with open(join(results_path, text_file), 'r') as f:
		content_list = [float(line) for line in f]

	# create new plot
	# plt.figure(figsize=(20, 10))
	plt.plot(content_list, plot_color + '-')

	# set plot title and labels
	x_label, y_label, model_no = filename.split('_')
	plt.title(('%s Over %s For Model No. %s' % (y_label, x_label, model_no)).title())
	plt.xlabel(x_label.title())
	plt.ylabel(y_label.title())

	# set limits of x to be outermost points
	plt.xlim([0, len(content_list) - 1])

	# remove old plot of file before saving new one
	[remove(join(plots_path, plot_file)) for plot_file in listdir(plots_path) if plot_file.startswith(filename)]

	# save new plot
	plt.savefig(join(plots_path, '%s_plot_%d.png' % (filename, time())))

	# clear and close plot
	plt.clf()
	plt.close()
