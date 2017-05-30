import pickle
from os.path import dirname, join
from math import ceil, floor

import numpy as np

file_path = dirname(__file__)

# open full meta set, contains entries of format tuple(image relative path, identification vector, expression vector)
with open(join(file_path, 'metadata', 'IMFDB_full_meta.pickle'), 'rb') as f:
	full_meta = pickle.load(f)

# split into training meta set, validation meta set and test meta set consisting of 70/20/10 percent of full meta set
train_size = ceil(len(full_meta) * 0.7)
val_size = floor(len(full_meta) * 0.2)
test_size = floor(len(full_meta) * 0.1)

# check that meta subsets constitute the full meta set
assert train_size + val_size + test_size == len(full_meta), \
	'Training meta set, validation meta set and test meta set sizes do not match with full meta set size'

# randomize order of entries in full meta set
np.random.shuffle(full_meta)

# extract meta subsets
train_meta = full_meta[:train_size]
val_meta = full_meta[train_size:train_size + val_size]
test_meta = full_meta[train_size + val_size:]

# save meta subsets in their respective pickle-file
with open(join(file_path, 'metadata', 'IMFDB_training_meta.pickle'), 'wb') as f:
	pickle.dump(train_meta, f)
with open(join(file_path, 'metadata', 'IMFDB_validation_meta.pickle'), 'wb') as f:
	pickle.dump(val_meta, f)
with open(join(file_path, 'metadata', 'IMFDB_testing_meta.pickle'), 'wb') as f:
	pickle.dump(test_meta, f)

print('\nTraining meta set, validation meta set and test meta set successfully created.')
