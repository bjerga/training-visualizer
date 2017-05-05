import numpy as np


# convert image from 3- or 4-dimensional to 2-dimensional
def process_image_dim(img):
	# image has correct number of dimensions
	if img.ndim == 2:
		return img
	# image has 3 dimensions, but is grayscale
	if img.shape[2] == 1:
		return img[:, :, 0]
	# image has 3 dimensions, and is rgb
	if img.shape[2] == 3:
		img = np.dstack([img, np.ones(img.shape[:2], np.uint8) * 255])
	img = np.squeeze(img.view(np.uint32))
	return img


def is_grayscale(img):
	if img.ndim == 2:
		return True
	if img.shape[2] == 1:
		return True
	return False


def add_image_from_array(fig, img):
	if is_grayscale(img):
		img = process_image_dim(img)
		fig.image(image=[img[::-1]], x=0, y=0, dw=img.shape[1], dh=img.shape[0])
	else:
		img = process_image_dim(img)
		fig.image_rgba(image=[img[::-1]], x=0, y=0, dw=img.shape[1], dh=img.shape[0])


def add_image_from_source(fig, source, img, img_name, add_to_source=True, always_grayscale = False):
	if is_grayscale(img) or always_grayscale:
		img = process_image_dim(img)
		fig.image(image=img_name, x=0, y=0, dw=img.shape[1], dh=img.shape[0], source=source)
	else:
		img = process_image_dim(img)
		fig.image_rgba(image=img_name, x=0, y=0, dw=img.shape[1], dh=img.shape[0], source=source)
	if add_to_source:
		source.add([img[::-1]], name=img_name)
