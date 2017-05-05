from os import listdir

from bokeh.io import curdoc
from bokeh.layouts import layout
from bokeh.models import ColumnDataSource, Div, Paragraph, Range1d

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

# get original image
images_folder = join(UPLOAD_FOLDER, user, file, 'images')
image_name = listdir(images_folder)[-1]  # TODO: throw error here
orig_img = np.array(Image.open(join(images_folder, image_name)))

grid = []

div = Div(text="<h3>Visualization of the saliency maps</h3>", width=500)
grid.append([div])

p = Paragraph(text="There seems to be no visualizations produced yet.", width=500)
grid.append([p])

# flip image to display correctly in coordinate system with placeholder
saliency_maps_source = ColumnDataSource(data=dict(image=[np.zeros((1, 1, 1))]))

image_height = orig_img.shape[0]
image_width = orig_img.shape[1]


def create_figure(title, x_range, y_range, tools="box_zoom, reset, save, pan"):
	fig = figure(tools=tools, x_range=x_range, y_range=y_range)
	fig.title.text = title
	fig.axis.visible = False
	fig.toolbar.logo = None
	return fig


# create plot for the original image
orig_img_fig = figure(title="Original Image", plot_width=250, plot_height=250, tools="box_zoom, reset, save, pan",
					outline_line_color="black", outline_line_width=3)
orig_img_fig.x_range = Range1d(0, image_width, bounds=(0, image_width))
orig_img_fig.y_range = Range1d(0, image_height, bounds=(0, image_height))
orig_img_fig.axis.visible = False

add_image_from_array(orig_img_fig, orig_img)

# create plot for the saliency maps image
saliency_fig = figure(title="Original Image", plot_width=250, plot_height=250, tools="box_zoom, reset, save, pan",
					outline_line_color="black", outline_line_width=3)
saliency_fig.x_range = orig_img_fig.x_range
saliency_fig.y_range = orig_img_fig.y_range
saliency_fig.axis.visible = False

add_image_from_source(saliency_fig, saliency_maps_source, orig_img, 'image', add_to_source=False)

grid.append([orig_img_fig, saliency_fig])


def update_data():
	try:
		with open(join(results_path, 'saliency_maps.pickle'), 'rb') as f:
			saliency_maps_data = pickle.load(f)
		p.text = ""
		img = process_image_dim(saliency_maps_data.astype('uint8'))
		saliency_maps_source.data['image'] = [img[::-1]]
	except FileNotFoundError:
		# if no visualization has been produced yet, simply skip visualization
		return

update_data()

document.add_root(layout(grid))
document.add_periodic_callback(update_data, 5000)
