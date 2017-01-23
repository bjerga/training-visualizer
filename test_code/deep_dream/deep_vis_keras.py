import numpy as np

import scipy.misc
import time
import os
import h5py
from scipy.ndimage.filters import gaussian_filter

from keras.models import Sequential
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout
from keras import backend as K
from keras.applications.vgg16 import VGG16

# VGG16 mean values
MEAN_VALUES = np.array([103.939, 116.779, 123.68]).reshape((3, 1, 1))

# path to the model weights file.
weights_path = 'vgg16_weights.h5'


# util function to convert a tensor into a valid image
def deprocess(x):

    x += MEAN_VALUES.transpose((1, 2, 0))  # Add VGG16 mean values

    # x = x[::-1, :, :]  # Change from BGR to RGB
    # x = x.transpose((1, 2, 0))  # Change from (Channel,Height,Width) to (Height,Width,Channel)

    x = np.clip(x, 0, 255).astype('uint8')  # clip in [0;255] and convert to int
    return x


# Creates a VGG16 model and load the weights if available
# (see https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3)
def VGG_16(w_path=None):
    vgg_model = Sequential()
    vgg_model.add(ZeroPadding2D((1, 1), input_shape=(3, 224, 224)))
    vgg_model.add(Convolution2D(64, 3, 3, activation='relu'))
    vgg_model.add(ZeroPadding2D((1, 1)))
    vgg_model.add(Convolution2D(64, 3, 3, activation='relu'))
    vgg_model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    vgg_model.add(ZeroPadding2D((1, 1)))
    vgg_model.add(Convolution2D(128, 3, 3, activation='relu'))
    vgg_model.add(ZeroPadding2D((1, 1)))
    vgg_model.add(Convolution2D(128, 3, 3, activation='relu'))
    vgg_model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    vgg_model.add(ZeroPadding2D((1, 1)))
    vgg_model.add(Convolution2D(256, 3, 3, activation='relu'))
    vgg_model.add(ZeroPadding2D((1, 1)))
    vgg_model.add(Convolution2D(256, 3, 3, activation='relu'))
    vgg_model.add(ZeroPadding2D((1, 1)))
    vgg_model.add(Convolution2D(256, 3, 3, activation='relu'))
    vgg_model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    vgg_model.add(ZeroPadding2D((1, 1)))
    vgg_model.add(Convolution2D(512, 3, 3, activation='relu'))
    vgg_model.add(ZeroPadding2D((1, 1)))
    vgg_model.add(Convolution2D(512, 3, 3, activation='relu'))
    vgg_model.add(ZeroPadding2D((1, 1)))
    vgg_model.add(Convolution2D(512, 3, 3, activation='relu'))
    vgg_model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    vgg_model.add(ZeroPadding2D((1, 1)))
    vgg_model.add(Convolution2D(512, 3, 3, activation='relu'))
    vgg_model.add(ZeroPadding2D((1, 1)))
    vgg_model.add(Convolution2D(512, 3, 3, activation='relu'))
    vgg_model.add(ZeroPadding2D((1, 1)))
    vgg_model.add(Convolution2D(512, 3, 3, activation='relu'))
    vgg_model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    vgg_model.add(Flatten())
    vgg_model.add(Dense(4096, activation='relu'))
    vgg_model.add(Dropout(0.5))
    vgg_model.add(Dense(4096, activation='relu'))
    vgg_model.add(Dropout(0.5))
    vgg_model.add(Dense(1000, activation='linear'))  # avoid softmax (see Simonyan 2013)

    if w_path:
        vgg_model.load_weights(w_path)

    return vgg_model


def built_in_vgg_16():
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


# Creates the VGG models and loads weights
# model = VGG_16(weights_path)
model = built_in_vgg_16()

# Specify input and output of the network
input_img = model.layers[0].input
layer_output = model.layers[-1].output

# List of the generated images after learning
kept_images = []

# Update coefficient
learning_rate = 1000.

# how many times we update image
no_of_iterations = 500

# specify L2-decay
l2_decay = 0.0001

# specify frequency of blurring and standard deviation for kernel for Gaussian blur
blur_freq = 4
blur_std = 0.3

# choose whether to include regularization
regularize = True

for class_index in [130, 351, 736, 850]:  # 130 flamingo, 351 hartebeest, 736 pool table, 850 teddy bear
    print('Processing filter %d' % class_index)
    start_time = time.time()

    # The loss is the activation of the neuron for the chosen class
    loss = layer_output[0, class_index]

    # we compute the gradient of the input picture w.r.t. this loss
    grads = K.gradients(loss, input_img)[0]

    # this function returns the loss and grads given the input picture
    # also add a flag to disable the learning phase (in our case dropout)
    iterate = K.function([input_img, K.learning_phase()], [loss, grads])

    np.random.seed(1337)  # for reproducibility
    # we start from a gray image with some random noise

    input_img_data = np.random.normal(0, 10, (1,) + model.input_shape[1:])  # (1,) for batch axis

    # we run gradient ascent for 1000 steps
    # for i in range(1000):
    if class_index == 130:
        for i in range(no_of_iterations):
            # input 0 for test phase
            loss_value, grads_value = iterate([input_img_data, 0])

            # Apply gradient to image
            input_img_data += grads_value * learning_rate

            if regularize:
                # apply L2-regularizer
                input_img_data *= (1 - l2_decay)

                # apply Gaussian blur
                if not i % blur_freq:
                    # blur along height and width, but not channels (colors)
                    input_img_data = gaussian_filter(input_img_data, sigma=[0, 0, blur_std, blur_std])

            # print('Current loss value:', loss_value)
            print('Round %d finished.' % i)

    # decode the resulting input image and add it to the list
    img = deprocess(input_img_data[0])
    kept_images.append((img, loss_value))
    end_time = time.time()
    print('Filter %d processed in %ds' % (class_index, end_time - start_time))

# Compute the size of the grid
n = int(np.ceil(np.sqrt(len(kept_images))))

# build a black picture with enough space for the kept_images
# img_height = model.input_shape[2]
# img_width = model.input_shape[3]
img_height = model.input_shape[1]
img_width = model.input_shape[2]
margin = 5
height = n * img_height + (n - 1) * margin
width = n * img_width + (n - 1) * margin
stitched_res = np.zeros((height, width, 3))

# fill the picture with our saved filters
for i in range(n):
    for j in range(n):
        if len(kept_images) <= i * n + j:
            break
        img, loss = kept_images[i * n + j]
        stitched_res[(img_height + margin) * i: (img_height + margin) * i + img_height,
        (img_width + margin) * j: (img_width + margin) * j + img_width, :] = img

# save the result to disk

result_name = 'output/'
if regularize:
    result_name += 'regularized_lr=%s_iter=%s_l2=%s_bf=%s_bstd=%s.png' \
                   % (str(learning_rate), str(no_of_iterations), str(l2_decay).replace('.', '-'),
                      str(blur_freq), str(blur_std).replace('.', '-'))
else:
    result_name += 'vanilla_lr=%s_iter=%s.png' % (str(learning_rate).replace('.', '-'), str(no_of_iterations))

scipy.misc.toimage(stitched_res, cmin=0, cmax=255).save(result_name)
# Do not use scipy.misc.imsave because it will normalize the image pixel value between 0 and 255
