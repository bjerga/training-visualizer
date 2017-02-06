from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np

import matplotlib.pyplot as plt


# https://github.com/Lasagne/Recipes/blob/master/examples/Saliency%20Maps%20and%20Guided%20Backpropagation.ipynb


def create_intermediate_model():
	base_model = VGG16(include_top=True, weights='imagenet')
	model = Model(input=base_model.input, output=base_model.layers[-1].input)
	return model


def create_model():
    base_model = VGG16(include_top=True, weights='imagenet')

    vgg_model = Sequential()

    # save weights from last layer (softmax)
    softmax_weights = base_model.layers[-1].get_weights()

    # remove softmax layer
    base_model.layers.pop()

    # add VGG base layers (add separately to avoid adding one, big layer)
    for layer in base_model.layers:
        vgg_model.add(layer)

    # add new, linear layer
    vgg_model.add(Dense(1000, activation='linear', weights=softmax_weights))

    return vgg_model


def process_input(img):
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0) # from (224, 224, 3) to (1, 224, 224, 3)
	x = preprocess_input(x)
	return x


def get_saliency_function(model_input, model_output, class_index):

	# The loss is the activation of the neuron for the chosen class
	loss = model_output[0, class_index]

	# we compute the gradient/saliency of the input picture w.r.t. this loss
	saliency = K.gradients(loss, model_input)[0]
	#saliency = K.gradients(model_output, model_input)[0]

	# this function returns the loss and grads given the input picture
	# also add a flag to disable the learning phase (in our case dropout)
	return K.function([model_input], [saliency])


'''def compile_saliency_function(model, inp, outp):

	# problem: shape på output passer ikke inn her (max og så sum..?)

	# problem: må ha inn faktisk input og output
	#inp = model.inputs[0]
	#outp = model.layers[-1].input
	max_outp = np.amax(outp)
	#saliency = K.gradients(inp, K.sum(max_outp))[0] # problem: returnerer NoneType
	saliency = K.gradients(inp, max_outp)[0]
	max_class = np.argmax(outp)

	print(type(inp))
	print(type(saliency))
	print(type(max_class))
	return

	return K.function([inp], [saliency, max_class])'''


def show_images(img_original, saliency, class_label):


	# convert saliency from BGR to RGB
	saliency = saliency[:, :, ::-1]
	# reset zero-centering by adding mean pixel <-- maybe not necessary?
	#saliency[:, :, 0] += 103.939
	#saliency[:, :, 1] += 116.779
	#saliency[:, :, 2] += 123.68


	# plot the original image and the three saliency map variants
	plt.figure(figsize=(10, 10), facecolor='w')
	plt.suptitle("Saliency maps for class: " + class_label)
	#plt.suptitle("Class: " + classes[max_class] + ". Saliency: " + title)
	
	plt.subplot(2, 2, 1)
	plt.title('input')
	plt.imshow(img_original)

	plt.subplot(2, 2, 2)
	plt.title('abs. saliency')
	# absolute saliency
	abs_saliency = np.abs(saliency)
	# convert from rgb to grayscale (take max of each RGB value)
	abs_saliency = np.amax(abs_saliency, axis=2)
	# reshape
	abs_saliency = np.expand_dims(abs_saliency, axis=3)
	# plot absolute value with gray colormap
	plt.imshow(image.array_to_img(abs_saliency), cmap='gray')

	
	plt.subplot(2, 2, 3)
	plt.title('pos. saliency')
	pos_saliency = np.maximum(0, saliency) / np.amax(saliency)
	plt.imshow(image.array_to_img(pos_saliency))
	#plt.imshow((np.maximum(0, saliency) / saliency.max()))

	plt.subplot(2, 2, 4)
	plt.title('neg. saliency')
	neg_saliency = np.maximum(0, -saliency) / -np.amin(saliency)
	plt.imshow(image.array_to_img(neg_saliency))
	#plt.imshow((np.maximum(0, -saliency) / -saliency.min()))'''
	
	plt.show()


def main():
	# Creates the VGG model without the softmax layer
	#model = create_intermediate_model()
	model = create_model()

	img_path = 'cat.jpg'
	img = image.load_img(img_path, target_size=(224, 224))

	x = process_input(img)

	preds = model.predict(x)
	max_class = np.argmax(preds)

	# compute gradient and saliency for the given class
	saliency_fn = get_saliency_function(model.input, model.output, max_class)
	saliency = saliency_fn([x])

	#base_model = VGG16(weights='imagenet', include_top=True)
	#class_preds = base_model.predict(x)

	show_images(img, saliency[0][0], decode_predictions(preds)[0][1])


main()
