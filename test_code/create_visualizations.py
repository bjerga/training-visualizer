import glob
import os
import pickle

import matplotlib.pyplot as plt

import math
import numpy as np

base_folder = "/home/anniea/Code/results/vgg_results"

results = glob.glob(os.path.join(base_folder, '*'))


def create_training_progress_vis(results_path):

	with open(os.path.join(results_path, 'training_progress.txt')) as f:
		training_progress_data = list(zip(*[line.strip().split() for line in f]))
	try:
		with open(os.pathjoin(results_path, 'training_progress_val.txt')) as f:
			validation_progress_data = list(zip(*[line.strip().split() for line in f]))
	except FileNotFoundError:
		# this means that no validation data has been created, set to empty
		validation_progress_data = []


def create_layer_activations_vis(results_path, save_folder):

	with open(os.path.join(results_path, "layer_activations.pickle"), "rb") as f:
		layer_activations_data = pickle.load(f)

	for layer_name, filters in layer_activations_data:

		postfix = "_".join(os.path.basename(results_path).split('_')[1:])
		temp = layer_name.split(':')[-1]
		save_path = os.path.join(save_folder, "layer_act_{}_{}.png".format(postfix, temp))

		print(layer_name)

		#print(filters.shape)
		if layer_name != 'Layer 2: block1_conv2':
			continue

		if len(filters.shape) == 3:

			no_of_images = len(filters)

			if no_of_images < 4:
				no_of_rows = 1
			elif no_of_images < 64:
				no_of_rows = 4
			else:
				no_of_rows = 8

			no_of_cols = math.ceil(no_of_images / no_of_rows)

			# line the filters up horizontally and pad them with white to separate the filters
			images = np.hstack([np.pad(f, 2, 'constant', constant_values=255) for f in filters])

			total_width = images.shape[1]

			if no_of_images > no_of_cols:
				step = math.ceil(total_width / no_of_rows)
				images = np.vstack([images[:, x:x + step] for x in range(0, total_width, step)])

		elif len(filters.shape) == 1:
			return
			images = filters[np.newaxis, :]

		plt.figure(figsize=(7, 7), facecolor='w')
		plt.axis('off')
		fig = plt.imshow(images, cmap='gray', interpolation="nearest", vmin=0, vmax=255)
		fig.axes.get_xaxis().set_visible(False)
		fig.axes.get_yaxis().set_visible(False)
		plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
		plt.close()


def create_saliency_maps_vis(results_path, save_folder):

	with open(os.path.join(results_path, "saliency_maps.pickle"), "rb") as f:
		saliency_maps_data = pickle.load(f)

	postfix = "_".join(os.path.basename(results_path).split('_')[1:])
	save_path = os.path.join(save_folder, "saliency_{}.png".format(postfix))

	plt.figure(figsize=(5, 5), facecolor='w')
	plt.axis('off')
	fig = plt.imshow(saliency_maps_data[:, :, 0], cmap='gray', interpolation="nearest", vmin=0, vmax=255)
	fig.axes.get_xaxis().set_visible(False)
	fig.axes.get_yaxis().set_visible(False)
	plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
	plt.close()


def create_deconvolution_network_vis(results_path, save_folder):

	with open(os.path.join(results_path, "deconvolutional_network.pickle"), "rb") as f:
		deconvolution_network_data = pickle.load(f)

	for array, layer_name, feat_map_no in deconvolution_network_data:

		background = [123, 116, 103]

		'''new_image = []

		for row in array:
			new_row = []
			for pixels in row:
				if np.all(pixels != background):
					new_row.append(pixels)
			if new_row:
				new_image.append(new_row)'''

		left = 0
		right = 100000
		top = 100000
		bottom = 0

		for i in range(array.shape[0]):
			for j in range(array.shape[1]):
				if np.all(array[i, j, :] != background):
					if i < top:
						top = i
					if i > bottom:
						bottom = i
					if j > left:
						left = j
					if j < right:
						right = j

		print("{} {} {} {}".format(left, right, top, bottom))

		array = array[top:bottom+1, right:left+1, :]
		#array = array[array != background]
		print(array.shape)

		postfix = "_".join(os.path.basename(results_path).split('_')[1:])
		save_path = os.path.join(save_folder, "deconvolution_{}_{}_{}.png".format(postfix, layer_name, feat_map_no))

		plt.figure(figsize=(5, 5), facecolor='w')
		plt.axis('off')
		#fig = plt.imshow(array[:, :, 0], cmap='gray', interpolation="nearest", vmin=0, vmax=255)
		fig = plt.imshow(array, interpolation="nearest", vmin=0, vmax=255)
		fig.axes.get_xaxis().set_visible(False)
		fig.axes.get_yaxis().set_visible(False)
		plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
		plt.close()


def create_deep_visualization_vis(results_path, save_folder):

	with open(os.path.join(results_path, "deep_visualization.pickle"), "rb") as f:
		deep_visualization_data = pickle.load(f)

	background = [123, 116, 103]
	background2 = [124, 116, 104]

	for array, layer_name, unit_index, loss_value in deep_visualization_data:
		left = 0
		right = 100000
		top = 100000
		bottom = 0

		for i in range(array.shape[0]):
			for j in range(array.shape[1]):
				if np.all(array[i, j, :] != background) and np.all(array[i, j, :] != background2):
					if i < top:
						top = i
					if i > bottom:
						bottom = i
					if j > left:
						left = j
					if j < right:
						right = j

		print("{} {} {} {}".format(left, right, top, bottom))

		array = array[top:bottom + 1, right:left + 1, :]
		print(array.shape)

		postfix = "_".join(os.path.basename(results_path).split('_')[1:])
		save_path = os.path.join(save_folder, "deepvis_{}_{}_{}.png".format(postfix, layer_name, unit_index))

		plt.figure(figsize=(5, 5), facecolor='w')
		plt.axis('off')
		#fig = plt.imshow(array[:, :, 0], cmap='gray', interpolation="nearest", vmin=0, vmax=255)
		fig = plt.imshow(array, interpolation="nearest", vmin=0, vmax=255)
		fig.axes.get_xaxis().set_visible(False)
		fig.axes.get_yaxis().set_visible(False)
		plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
		plt.close()


def main():
	save_folder = "/home/anniea/Code/result_images_vgg"

	for results_path in results:
		print(results_path)

		'''print("layer activations...")
		# layer activations
		create_layer_activations_vis(results_path, save_folder)

		print("saliency maps...")
		# saliency maps
		create_saliency_maps_vis(results_path, save_folder)

		print("deconvolution...")
		# deconvolution
		create_deconvolution_network_vis(results_path, save_folder)'''

		print("deep visualization...")
		# deep visualization
		create_deep_visualization_vis(results_path, save_folder)


main()
