import os
import glob
import pickle
from PIL import Image

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

for actor_path in actors:
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

print(len(training_data))

#with open(os.path.join('/Users/annieaa/Documents/NTNU/Fordypningsprosjekt/visualizer/case_study', 'imfdb_training_data.pickle'), 'wb') as f:
#	pickle.dump(training_data, f)




