from os import listdir

from bokeh.io import curdoc
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource, Paragraph, Range1d, BoxZoomTool, Column

from os.path import join

from bokeh.plotting import figure
from PIL import Image
import pickle

from visualizer.config import UPLOAD_FOLDER, UPDATE_INTERVALS
from custom_bokeh.utils import *

document = curdoc()

args = document.session_context.request.arguments

try:
	file = args['file'][0].decode('ascii')
	user = args['user'][0].decode('ascii')
except KeyError as e:
	raise KeyError(str(e) + '. Filename and username must be provided as request parameters.')

# find path for result data
results_path = join(UPLOAD_FOLDER, user, file, 'results')

grid = []

p = Paragraph(text="", width=500)
layout = Column(children=[p])

saliency_maps_source = ColumnDataSource(data=dict())


def fill_data_source(saliency_maps_data):

	img_width = saliency_maps_data.shape[0]
	img_height = saliency_maps_data.shape[1]

	# get original image
	try:
		images_folder = join(UPLOAD_FOLDER, user, file, 'images')
		image_name = listdir(images_folder)[-1]
		orig_img = np.array(Image.open(join(images_folder, image_name)).resize((img_width, img_height)))
	except FileNotFoundError:
		raise FileNotFoundError(str(e) + '. No visualization input image found in the image folder')

	# create plot for the original image
	orig_img_fig = figure(title="Original Image", plot_width=250, plot_height=250, tools="pan, reset, save",
							x_range=Range1d(0, img_width, bounds=(0, img_width)),
							y_range=Range1d(0, img_height, bounds=(0, img_height)),
							outline_line_color="black", outline_line_width=3)
	orig_img_fig.add_tools(BoxZoomTool(match_aspect=True))
	orig_img_fig.axis.visible = False
	orig_img_fig.toolbar.logo = None

	add_image_from_array(orig_img_fig, orig_img)

	# create plot for the saliency maps image
	saliency_fig = figure(title="Absolute Saliency", plot_width=250, plot_height=250, tools="pan, reset, save",
							x_range=orig_img_fig.x_range, y_range=orig_img_fig.y_range,
							outline_line_color="black", outline_line_width=3)
	saliency_fig.add_tools(BoxZoomTool(match_aspect=True))
	saliency_fig.axis.visible = False
	saliency_fig.toolbar.logo = None

	add_image_from_source(saliency_fig, saliency_maps_source, saliency_maps_data, 'abs_saliency', always_grayscale=True)

	layout.children.append(gridplot([orig_img_fig, saliency_fig], ncols=2, merge_tools=False))


def update_data():
	try:
		with open(join(results_path, 'saliency_maps.pickle'), 'rb') as f:
			saliency_maps_data = pickle.load(f)

		if not saliency_maps_source.data:
			# temporary remove callback to make sure the function is not being called while creating the visualizations
			document.remove_periodic_callback(update_data)
			fill_data_source(saliency_maps_data)
			document.add_periodic_callback(update_data, UPDATE_INTERVALS['saliency_maps'])
		else:
			# if the data source already exists, we can simply update its data
			img = process_image_dim(saliency_maps_data.astype('uint8'))
			# dictionary that holds new saliency map
			new_saliency_maps_data = dict(
				abs_saliency=[img[::-1]]
			)
			saliency_maps_source.data = new_saliency_maps_data

		p.text = ""

	except FileNotFoundError:
		p.text = "There are no visualization data produced yet."
		# if no visualization has been produced yet, simply skip visualization
		return

document.add_root(layout)
document.add_periodic_callback(update_data, UPDATE_INTERVALS['saliency_maps'])
