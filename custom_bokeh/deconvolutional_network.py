from os import listdir

from bokeh.io import curdoc
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource, Div, Paragraph, Column, Range1d, BoxZoomTool

from os.path import join

from bokeh.plotting import figure
from PIL import Image
import pickle

from visualizer.config import UPLOAD_FOLDER
from custom_bokeh.helpers import *

document = curdoc()
args = document.session_context.request.arguments

# TODO: throw error if these are not provided
file = args['file'][0].decode('ascii')
user = args['user'][0].decode('ascii')

# find path for result data
results_path = join(UPLOAD_FOLDER, user, file, 'results')

deconvolution_source = ColumnDataSource(data=dict())

p = Paragraph(text="", width=500)
layout = Column(children=[p])


def fill_data_source(deconvolution_data):
	p.text = "Visualizations are being produced..."

	img_width = deconvolution_data[0][0].shape[0]
	img_height = deconvolution_data[0][0].shape[1]

	# get original image
	images_folder = join(UPLOAD_FOLDER, user, file, 'images')
	image_name = listdir(images_folder)[-1]  # TODO: throw error here
	orig_img = np.array(Image.open(join(images_folder, image_name)).resize((img_width, img_height)))


	# create plot for the original image
	orig_img_fig = figure(title="Original Image", plot_width=250, plot_height=250, tools="reset, save, pan",
						  x_range=Range1d(0, img_width, bounds=(0, img_width)),
						  y_range=Range1d(0, img_height, bounds=(0, img_height)),
						  outline_line_color="black", outline_line_width=3)
	orig_img_fig.add_tools(BoxZoomTool(match_aspect=True))
	orig_img_fig.axis.visible = False
	orig_img_fig.toolbar.logo = None

	add_image_from_array(orig_img_fig, orig_img)

	layout.children.append(orig_img_fig)

	figures = []

	# loop through deconvolution data
	for array, layer_name, feat_map_no in deconvolution_data:

		name = "{}_{}".format(layer_name, feat_map_no)
		title = "#{} in {}".format(feat_map_no, layer_name)

		fig = figure(title=title, tools="reset, save, pan", plot_width=250, plot_height=250, outline_line_color="black",
						outline_line_width=3, x_range=orig_img_fig.x_range, y_range=orig_img_fig.y_range)
		fig.add_tools(BoxZoomTool(match_aspect=True))
		fig.axis.visible = False
		fig.toolbar.logo = None
		fig.min_border_left = 20

		add_image_from_source(fig, deconvolution_source, array, name)

		figures.append(fig)

	# make a grid of the feature maps
	grid = gridplot(figures, ncols=4, toolbar_options=dict(logo=None))

	layout.children.append(Div(text="<div align='center'><b>Feature Map Visualizations</b></div>", width=1000))
	layout.children.append(grid)
	p.text = ""


def update_data():
	global create_source
	try:
		# load deconvolution data, on form tuple(image array, layer name, feat map number)
		with open(join(results_path, 'deconvolutional_network.pickle'), 'rb') as f:
			deconvolution_data = pickle.load(f)

		# if it is the first time data is detected, we need to fill the data source with the images
		if not deconvolution_source.data:
			# temporary remove callback to make sure the function is not being called while creating the visualizations
			document.remove_periodic_callback(update_data)
			fill_data_source(deconvolution_data)
			document.add_periodic_callback(update_data, 5000)
		# if not, we can simply update the data
		else:
			for array, layer_name, feat_map_no in deconvolution_data:
				name = "{}_{}".format(layer_name, feat_map_no)

				img = process_image_dim(array)
				deconvolution_source.data[name] = [img[::-1]]

		p.text = ""
	except FileNotFoundError:
		p.text = "There are no visualization data produced yet."
		# this means deconvolution data has not been created yet, skip visualization
		return
	except EOFError:
		p.text = "There are no visualization data produced yet."
		# this means deconvolution data has been created, but is empty, skip visualization
		return


document.add_root(layout)
document.add_periodic_callback(update_data, 5000)
update_data()
