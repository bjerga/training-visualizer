import os
import glob
import pickle
from PIL import Image
import math

# TODO: this whole file should be updated to be cleaner

root = '/Users/annieaa/Documents/NTNU/Fordypningsprosjekt/IMFDB'

expressions = ['NEUTRAL', 'ANGER', 'HAPPINESS', 'SADNESS', 'SURPRISE', 'FEAR', 'DISGUST']
expression_dict = {'NEUTRAL': 0, 'ANGER': 1, 'HAPPINESS': 2, 'SADNESS': 3, 'SURPRISE': 4, 'FEAR': 5, 'DISGUST': 6}

actors = glob.glob(os.path.join(root, '*'))

'''with open(os.path.join('/Users/annieaa/Documents/NTNU/Fordypningsprosjekt/IMFDB/Savithri/Missamma', 'Missamma.txt'), 'r') as f:
	data = f.read()

data = data.replace('Savithri_', 'Savitri_')

with open(os.path.join('/Users/annieaa/Documents/NTNU/Fordypningsprosjekt/IMFDB/Savithri/Missamma', 'Missamma.txt'), 'w') as f:
	f.write(data)'''

training_data = []

counter = 0

'''for actor_path in actors:
	actor = os.path.basename(actor_path)
	movies = glob.glob(os.path.join(actor_path, '*'))
	for movie_path in movies:
		movie = os.path.basename(movie_path)
		with open(os.path.join(movie_path, movie + '.txt')) as f:
			for line in f.readlines():
				entries = line.strip().split()
				if len(entries) > 1:
					img_name = entries[2]
					img_path = os.path.join(movie_path, 'images', img_name)
					img_rel_path = os.path.relpath(img_path, start='/Users/annieaa/Documents/NTNU/Fordypningsprosjekt')
					expression = entries[11]
					expression_vector = [0 for _ in range(len(expressions))]
					expression_vector[expression_dict[expression]] = 1
					actor_vector = [0 for _ in range(len(actors))]
					actor_vector[counter] = 1

					training_data.append((img_rel_path, actor_vector, expression_vector))

					#image = Image.open(img_path)

					#print("{}: {} --> {}".format(img_path, expression, expression_vector))
	counter += 1

print(len(training_data))'''

#with open(os.path.join('/Users/annieaa/Documents/NTNU/Fordypningsprosjekt/visualizer/case_study', 'imfdb_training_data.pickle'), 'wb') as f:
#	pickle.dump(training_data, f)

# AVERAGES:

# print(widths/number_of_images)  # 101.85763185005483
# print(heights/number_of_images)  # 128.47163603735336
# print(min_width)  # 8
# print(min_height)  # 11
# print(max_width)  # 852
# print(max_height)  # 964


base_url = '/Users/annieaa/Documents/NTNU/Fordypningsprosjekt'

with open(os.path.join(base_url, 'imfdb_training_data.pickle'), 'rb') as f:
	training_data = pickle.load(f)

img_size = (130, 130)


def get_averages():

	r, g, b = 0, 0, 0
	number_of_pixels = 0

	number_of_images = 0
	widths = 0
	heights = 0
	min_width = 10000
	min_height = 10000
	max_width = 0
	max_height = 0

	for x in training_data:
		number_of_images += 1
		print(number_of_images)
		image = Image.open(os.path.join(base_url, x[0]))
		#width, height = image.size

		pixels = list(image.getdata())

		number_of_pixels += len(pixels)

		pixels = list(zip(*pixels))

		r += sum(pixels[0])
		g += sum(pixels[1])
		b += sum(pixels[2])

		'''if height < min_height:
			min_height = height
		if width < min_width:
			min_width = width
		if height > max_height:
			max_height = height
		if width > max_width:
			max_width = width

		widths += width
		heights += height'''


	print("{} pixels".format(number_of_pixels))
	print("{} red, {} green, {} blue".format(r, g, b))
	print("Average: {} red, {} green, {} blue".format(r/number_of_pixels, g/number_of_pixels, b/number_of_pixels))


def pad_images():
	for x in training_data:
		path = os.path.join(base_url, x[0])
		image = Image.open(path)
		width, height = image.size

		if width < height:
			new_width = math.ceil(img_size[1] * width / height)
			image = image.resize((new_width, img_size[1]), Image.ANTIALIAS)
		elif width > height:
			new_height = math.ceil(img_size[0] * height / width)
			image = image.resize((img_size[0], new_height), Image.ANTIALIAS)
		else:
			image = image.resize(img_size)

		new_image = Image.new('RGB', img_size)
		new_image.paste(image, (math.ceil((img_size[0] - image.size[0])/2),
								math.ceil((img_size[1] - image.size[1])/2)))
		new_image.save(path)



get_averages()
