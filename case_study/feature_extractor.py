import pickle
from os.path import join

import numpy as np
from PIL import Image

import keras.backend as K
from keras.preprocessing.image import img_to_array

from keras_vggface.vggface import VGGFace

# BGR mean values from GitHub repo of keras_vggface
MEAN_VALUES = np.array([93.5940, 104.7624, 129.1863])

# set paths to case study file and data
case_path = '/home/mikaelbj/Documents/GitHub/training-visualizer/case_study'
data_path = '/home/mikaelbj/Documents/case_study_data'

# collect meta data files for images
with open(join(case_path, 'metadata', 'IMFDB_training_meta.pickle'), 'rb') as f:
	training_meta = pickle.load(f)
with open(join(case_path, 'metadata', 'IMFDB_validation_meta.pickle'), 'rb') as f:
	validation_meta = pickle.load(f)
with open(join(case_path, 'metadata', 'IMFDB_testing_meta.pickle'), 'rb') as f:
	testing_meta = pickle.load(f)

# make lists to hold new meta with features instead of image names
metadata_feat_list = [[], [], []]

# select names for meta with features
name_base = 'IMFDB_{}_meta_feat.pickle'
metadata_feat_names = [name_base.format('training'), name_base.format('validation'), name_base.format('testing')]

# get original VGGFace model
model = VGGFace(include_top=True, input_shape=(224, 224, 3))

# define function to get input to last layer (layers[-1] is activation layer)
get_features = K.function([model.input, K.learning_phase()], [model.layers[-2].input])

metadata_list = [training_meta, validation_meta, testing_meta]
for i in range(len(metadata_list)):

	metadata = metadata_list[i]
	for img_rel_path, id_vector, expression_vector in metadata:
		img = Image.open(join(data_path, img_rel_path))
		img = img.resize((224, 224))
		img_array = img_to_array(img)

		# alter to BGR and subtract mean values
		if K.image_data_format() == 'channels_last':
			img_array = img_array[:, :, ::-1]
			img_array -= MEAN_VALUES.reshape((1, 1, 3))
		else:
			img_array = img_array[::-1, :, :]
			img_array -= MEAN_VALUES.reshape((3, 1, 1))

		metadata_feat_list[i].append((get_features([[img_array], 0])[0][0], id_vector, expression_vector))

	# save meta with features
	with open(join(case_path, 'metadata_feat', metadata_feat_names[i]), 'wb') as f:
		pickle.dump(metadata_feat_list[i], f)

	# empty list to as content is no longer needed
	metadata_feat_list[i] = []

	print(metadata_feat_names[i], 'finished processing')
