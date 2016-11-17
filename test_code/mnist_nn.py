import os
import struct
import pickle
import numpy as np
import matplotlib.pyplot as plt
from time import time
from array import array


file_path = os.path.dirname(__file__)
network_path = os.path.join(file_path, 'networks')
figure_path = os.path.join(file_path, 'figures')

# create necessary folders if they don't exist
try:
	os.mkdir(network_path)
except FileExistsError:
	pass

try:
	os.mkdir(figure_path)
except FileExistsError:
	pass


# activation function for hidden units
def relu(value):
	return np.maximum(0, value)


# derivative of activation function for hidden units
def d_relu(value):
	# change to (value > 0.0) + 0.0 if any problems
	return value > 0.0


# activation function for output units
def softmax(value):
	exponent = np.exp(value - np.amax(value))
	normalized_exponent = exponent / np.sum(exponent)
	return normalized_exponent


# reads data sets and returns a list of tuples on the form (image, label), where
# image is an one-dimensional array of pixel values in range [0.0, 1.0] and label is a number
def load_data(data_set):
	if data_set == 'training':
		label_filename = 'train-labels.idx1-ubyte'
		image_filename = 'train-images.idx3-ubyte'
	elif data_set == 'testing':
		label_filename = 't10k-labels.idx1-ubyte'
		image_filename = 't10k-images.idx3-ubyte'
	else:
		print('Invalid name for data set in loader')
	
	# specific to file-structure at server
	data_path = os.path.join(file_path.split('programs', 1)[0], 'data')
	label_file_path = os.path.join(data_path, label_filename)
	image_file_path = os.path.join(data_path, image_filename)
	
	# _ denotes unused variables
	# read labels
	label_file = open(label_file_path, 'rb')
	_, label_amount = struct.unpack('>II', label_file.read(8))
	labels = array('b', label_file.read())
	label_file.close()
	
	# read images
	image_file = open(image_file_path, 'rb')
	_, image_amount, rows, cols = struct.unpack('>IIII', image_file.read(16))
	images = array('B', image_file.read())
	image_file.close()
	
	# find size of images
	image_size = rows*cols
	
	# create a list of tuples (image, label) and
	# reduce range from [0, 255] to [0.0, 1.0]
	data = []
	for i in range(image_amount):
		data.append((np.array(images[i*image_size:(i+1)*image_size])/255.0, labels[i]))
	
	return data


# implementation of a shallow neural network for classification of handwritten numbers
# trained on MNIST data set
class NeuralNetwork:
	# hyper-parameters
	# can be altered to observe different results (low learning rate for ReLU)
	learning_rate = 0.01
	dropout = 0.5
	
	# initialize network with specified number of input units, hidden units and output units
	# weights are set to a random number, using normal distribution for hidden weights and uniform distribution
	# for output weights. these are modified with np.sqrt() to normalize the variance of the random numbers generated
	# in weight matrices, row X holds all connections to previous unit X,
	# e. g. row 3 in output_weights holds all weights connected to hidden unit 3
	def __init__(self, input_amount, hidden_amount, output_amount):
		self.input_amount = input_amount
		self.hidden_weights = np.random.randn(input_amount, hidden_amount) * np.sqrt(2.0 / input_amount)
		self.hidden = np.ones(hidden_amount)
		self.output_weights = np.random.rand(hidden_amount, output_amount) / np.sqrt(hidden_amount)
		self.output = np.ones(output_amount)
		
		# error measurements
		self.average_cross_entropy = []
		self.training_error = []
		self.test_error = []
	
	# used for training network with all data for a given number of epochs
	# training is on-line
	def train(self, data, epochs):
		
		print('Training commenced')
		
		# load test data for determining test error after each epoch
		test_data = load_data('testing')
		
		# use to determine and save the best weights based on test set error
		best_test_error = 100.0
		
		# prevent shuffle to affect data outside function
		data = np.copy(data)
		
		# used to avoid repeatedly using len()
		data_amount = len(data)
		
		total_rounds = 0
		for epoch in range(epochs):
			
			# used to calculate average cross entropy error
			cross_entropy_sum = 0
			
			# every epoch, train on-line with all available data
			np.random.shuffle(data)
			for i in range(data_amount):
				
				# classify current image with training-mode enabled (enables dropout-regularization)
				classification = self.classify(data[i][0], mode='training')
				
				# create target output using image label
				target = np.zeros(len(self.output))
				target[data[i][1]] = 1
				
				# prints every 10000 rounds, to monitor progress of training
				if i % 10000 == 0:
					print('Starting round ' + str(total_rounds) + ', ' + str(data_amount * epochs - total_rounds) +
						  ' left (' + format(total_rounds * 100.0 / (data_amount * epochs), '.2f') + '% finished)')
					total_rounds += 10000
				
				# calculate error and back-propagate
				output_error = (classification - target)
				hidden_error = d_relu(self.hidden) * np.dot(self.output_weights, output_error)
				
				# perform weight update
				self.output_weights -= self.learning_rate * output_error * np.reshape(self.hidden, (len(self.hidden), 1))
				self.hidden_weights -= self.learning_rate * hidden_error * np.reshape(data[i][0], (len(data[i][0]), 1))
				
				# add current cross entropy error
				# simplified from -sum(target * log(classification)) as target is non-zero for only one entry
				cross_entropy_sum += -np.log(classification[data[i][1]])
			
			print('\nEpochs completed: %d\n' % (epoch+1))
			
			# halve learning rate every 5 epochs to learn more fine-tuned weights in later epochs
			if epoch % 5 == 0 and epoch != 0:
				self.learning_rate -= self.learning_rate / 2.0
			
			# log error after each epoch, used to show progression
			self.average_cross_entropy.append(cross_entropy_sum / data_amount)
			self.training_error.append(self.test(data))
			self.test_error.append(self.test(test_data))
			
			# write errors to file, used to continuously plot progression
			# self.write_error_file('cross_entropy', self.average_cross_entropy)
			self.write_error_file('training_error', self.training_error)
			# self.write_error_file('test_error', self.test_error)
			
			# check if current weight produce lower test set error than best test set error
			# if true, save current weights as best weights so far
			if self.test_error[-1] < best_test_error:
				best_hidden_weights = self.hidden_weights
				best_output_weights = self.output_weights
		
		# training is completed, now set weights to the best weights found
		# deactivated in final version to choose best network based on average cross entropy
		# (if activated, printing of errors must be altered)
		# self.hidden_weights = best_hidden_weights
		# self.output_weights = best_output_weights
		
		# count number of saved networks, used for naming figures
		figure_nr = len(os.listdir(network_path))
		
		# plot training curve
		plt.plot(self.average_cross_entropy, 'g-', label='Average cross entropy error')
		plt.legend(loc='upper right')
		plt.title('Training curve')
		plt.xlabel('Epochs (60000 iterations per)')
		plt.ylabel('Error')
		plt.savefig(os.path.join(figure_path, 'nn' + str(figure_nr) + '_ace_error.png'))
		plt.clf()
		
		# plot training set and test set error
		plt.plot(self.training_error, 'b-', label='Training set error')
		plt.plot(self.test_error, 'r-', label='Test set error')
		plt.legend(loc='upper right')
		plt.title('Classification error')
		# plt.xlim(0, epochs)
		# plt.ylim(0.0, 0.5)
		plt.xlabel('Epochs (60000 iterations per)')
		plt.ylabel('Error (% misclassified)')
		plt.savefig(os.path.join(figure_path, 'nn' + str(figure_nr) + '_classification.png'))
		plt.clf()
		
		print('Training complete')
	
	# used for classifying an image, return a vector with probabilities for each number
	# mode used for enabling regularization with dropout when training
	def classify(self, image, mode='testing'):
		
		# compute output from hidden units
		self.hidden = relu(np.dot(np.transpose(self.hidden_weights), image))
		
		# if in training mode, enable inverted dropout
		# currently deactivated as it worsens results on both training set and test set (probable correlation)
		# if mode == 'training':
		#     drop_mask = (np.random.rand(len(self.hidden)) < self.dropout) / self.dropout
		#     self.hidden *= drop_mask
		
		# compute and return output from output units
		return softmax(np.dot(np.transpose(self.output_weights), self.hidden))
	
	# counts number of incorrect classifications for a given data set
	def test(self, data):
		
		incorrectly_classified = 0
		for i in range(len(data)):
			classification = self.classify(data[i][0])
			
			# if best probability in classification is the different from label,
			# image was incorrectly classified
			if classification.argmax() != data[i][1]:
				incorrectly_classified += 1
		
		# return % incorrectly classified
		return incorrectly_classified * 100.0 / len(data)
	
	# save network for later use / easy demonstration of correctness
	def save_network(self):
		# count number of saved networks, used for naming network
		nn_amount = len(os.listdir(network_path))
		
		# put relevant info in dictionary
		network_info = {'input_amount': self.input_amount,
						'hidden_amount': len(self.hidden),
						'output_amount': len(self.output),
						'output_weights': self.output_weights,
						'hidden_weights': self.hidden_weights,
						'average_cross_entropy': self.average_cross_entropy,
						'training_error': self.training_error,
						'test_error': self.test_error}
		
		# save network to given folder with unused name
		save_path = os.path.join(network_path, 'nn' + str(nn_amount) + '.pkl')
		with open(save_path, 'wb') as file:
			pickle.dump(network_info, file, 2)
		
		print('\nNetwork saved as nn' + str(nn_amount) + '.pkl')
	
	# load network for use / demonstration
	@staticmethod
	def load_network(network_number):
		
		# load network with given network number
		load_path = os.path.join(network_path, 'nn' + str(network_number) + '.pkl')
		with open(load_path, 'rb') as file:
			network_info = pickle.load(file)
		
		# create new network with saved arguments
		network = NeuralNetwork(network_info['input_amount'],
								network_info['hidden_amount'],
								network_info['output_amount'])
		
		# used saved information to recreate saved network
		network.hidden_weights = network_info['hidden_weights']
		network.output_weights = network_info['output_weights']
		network.average_cross_entropy = network_info['average_cross_entropy']
		network.training_error = network_info['training_error']
		network.test_error = network_info['test_error']
		
		# print relevant information about loaded network
		print('\nNetwork nn' + str(network_number) + ' loaded')
		print('Network has ' + str(network_info['input_amount']) + ' input units, ' +
			  str(network_info['hidden_amount']) + ' hidden units, and ' +
			  str(network_info['output_amount']) + ' output units')
		print('Network was trained for ' + str(len(network.average_cross_entropy)) + ' epochs')
		print('Average cross entropy error for previous training of nn' + str(network_number) + ' is ' +
			  str(network.average_cross_entropy[-1]))
		
		return network
	
	# write error results to txt-file
	@staticmethod
	def write_error_file(error_type, error_list):
		
		write_path = os.path.join(file_path, 'results', '%s.txt' % error_type)
		
		# overwrite old file if exists
		with open(write_path, 'w') as f:
			f.write('')
		# write each observation on a new line
		with open(write_path, 'a') as f:
			for error in error_list:
				f.write('%f\n' % error)  # python will convert \n to os.linesep


# used to create and run neural network
def main():
	# used to compute computation time
	start = time()
	
	# choose whether to train a new network or load an old one
	# -1 for training, desired network number for loading (e.g. 5 loads nn5.pkl)
	network_number = -1
	
	if network_number >= 0:
		
		# load network with specified number
		network = NeuralNetwork.load_network(network_number)
	
	else:
		
		# create network with desired number of input units, hidden units and output units
		network = NeuralNetwork(784, 100, 10)
		
		# select number of desired epochs and commence training
		epochs = 5
		train_data = load_data('training')
		network.train(train_data, epochs)
		
		# only save decent networks
		if network.test_error[-1] <= 2.50:
			network.save_network()
	
	# print information about classification error, using the latest results found in training
	# create a tuple containing size of training data set and test data set (used in printing)
	data_sizes = (60000, 10000)
	print('\nFor training data, network correctly classifies ' + format((100.0 - network.training_error[-1]), '.2f') +
		  '% of images, for a total of ' + format((100.0 - network.training_error[-1]) * data_sizes[0] / 100,
												  '.0f') + ' images')
	print('\nFor test data, network correctly classifies ' + format((100.0 - network.test_error[-1]), '.2f') +
		  '% of images, for a total of ' + format((100.0 - network.test_error[-1]) * data_sizes[1] / 100,
												  '.0f') + ' images')
	
	# print('\nBest test set error: ' + str(min(network.test_error)))
	
	# print total computation time
	print('\nComputation time was ' + format(time() - start, '.2f') + ' seconds')


main()
