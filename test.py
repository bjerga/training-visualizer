from PIL import Image
import numpy as np
import pickle
from os.path import join

url = '/Users/annieaa/Documents/NTNU/Fordypningsprosjekt/visualizer/visualizer/static/user_storage/anniea/vgg16_keras/results'

'''with open(join(url, 'saliency_maps.pickle'), 'rb') as f:
	saliency_maps_data = pickle.load(f)

img = Image.fromarray(saliency_maps_data.astype('uint8'))
img.show()'''


with open(join(url, 'deep_visualization.pickle'), 'rb') as f:
	deep_vis_data = pickle.load(f)

img = Image.fromarray(deep_vis_data[2][0])
img.show()