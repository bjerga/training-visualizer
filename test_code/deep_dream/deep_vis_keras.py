import numpy as np

import scipy.misc
from scipy.ndimage.filters import gaussian_filter
from os import mkdir
from os.path import dirname, join
from time import time

from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K
from keras.applications.vgg16 import VGG16

# define output path and make folder
output_path = join(dirname(__file__), 'output')
try:
    mkdir(output_path)
except FileExistsError:
    # folder exists, which is what we wanted
    pass

# TODO: delete when done with testing
is_VGG16 = True
VGG16_MEAN_VALUES = np.array([103.939, 116.779, 123.68])

# Update coefficient
learning_rate = 2500.

# how many times we update image
no_of_iterations = 500

# specify L2-decay
# used to prevent a small number of extreme pixel values from dominating the output image
l2_decay = 0.0001

# specify frequency of blurring and standard deviation for kernel for Gaussian blur
# used to penalize high frequency information in the output image
blur_interval = 4
# standard deviation values between 0.0 and 0.3 work poorly, according to yosinski
blur_std = 1.0

# specify value percentile limit
# used to induce sparsity by setting pixels with small absolute value to 0
value_percentile = 0

# specify norm percentile limit
# used to induce sparsity by setting pixels with small norm to 0
norm_percentile = 0

# specify contribution percentile limit
# used to induce sparsity by setting pixels with small contribution to 0
contribution_percentile = 0

# specify absolute contribution percentile limit
# used to induce sparsity by setting pixels with small absolute contribution to 0
abs_contribution_percentile = 0

# choose whether to include regularization
regularize = True

# TODO: update to Keras 2.0
# TODO: add support for greyscale images

# utility function used to convert an array into a savable image
def deprocess(vis_array, ch_dim):

    # remove batch dimension, and alter color dimension accordingly
    img_array = vis_array[0]
    ch_dim -= 1

    if is_VGG16:
        # create shape with correct color dimension
        new_shape = [1, 1, 1]
        new_shape[ch_dim] = 3
    
        # add VGG16 mean values
        img_array += VGG16_MEAN_VALUES.reshape(new_shape)

    if ch_dim == 0:
        # alter dimensions from (color, height, width) to (height, width, color)
        img_array = img_array.transpose((1, 2, 0))

    # clip in [0, 255], and convert to uint8
    img_array = np.clip(img_array, 0, 255).astype('uint8')
    
    return img_array


# creates a model to generate gradients from
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


# saves the visualization and a txt-file describing its creation environment
def save_visualization(img, class_index, loss_value):
    
    # create appropriate name to identify image
    if regularize:
        img_name = 'regularized'
    else:
        img_name = 'vanilla'
    img_name += '_{}_{}'.format(class_index, time())

    # save the resulting image to disk
    # avoid scipy.misc.imsave because it will normalize the image pixel value between 0 and 255
    scipy.misc.toimage(img, cmin=0, cmax=255).save(join(output_path, img_name + '.png'))

    # also save a txt-file containing information about creation environment and obtained loss
    img_info = 'Learning rate: {}\n' \
               'Number of iterations: {}\n' \
               '----------\n' \
               ''.format(learning_rate, no_of_iterations)
    if regularize:
        img_info += 'L2-decay: {}\n' \
                    'Blur interval and std: {} & {}\n' \
                    'Value percentile: {}\n' \
                    'Norm percentile: {}\n' \
                    'Contribution percentile: {}\n' \
                    'Abs. contribution percentile: {}\n' \
                    ''.format(l2_decay, blur_interval, blur_std, value_percentile, norm_percentile,
                              contribution_percentile, abs_contribution_percentile)
    img_info += '----------\n' \
                'Obtained loss value: {}\n' \
                ''.format(loss_value)
    with open(join(output_path, img_name + '_info.txt'), 'w') as f:
        f.write(img_info)
        
    print('\nImage of class {} have been saved as {}.png\n'.format(class_index, img_name))


# returns the function to easily compute the input image gradients w.r.t. the activations
def get_gradient_function(model_input, model_output, class_index):
    
    # loss is the activation of the neuron for the chosen class
    loss = model_output[0, class_index]
    
    # gradients are computed from the visualization w.r.t. this loss
    gradients = K.gradients(loss, model_input)[0]
    
    # return function returning the loss and gradients given a visualization image
    # add a flag to disable the learning phase (e.g. when using dropout)
    return K.function([model_input, K.learning_phase()], [loss, gradients])


# creates an random, initial image to manipulate into a visualization
def create_initial_image(model_input_shape):
    
    # TODO: remove when done with testing
    # set random seed to be able to reproduce initial state of image
    # used in testing only, and should be remove upon implementation with tool
    np.random.seed(1337)

    # set channel dimension based on image data format from Keras backend
    if K.image_data_format() == 'channels_last':
        ch_dim = 3
    else:
        ch_dim = 1
    
    # TODO: why 0 and 10 values? just for testing? check it out
    # return a random, initial image, and the channel (color) dimension of the image
    # add (1,) for batch dimension
    return np.random.normal(0, 10, (1,) + model_input_shape[1:]), model_input_shape[ch_dim]


# regularizes input image with various techniques
# each technique is activated by non-zero values for their respective global variables
def apply_ensemble_regularization(visualization, ch_dim, pixel_gradients, iteration):
    
    # regularizer #1
    # apply L2-decay
    if l2_decay > 0:
        visualization *= (1 - l2_decay)

    # regularizer #2
    # apply Gaussian blur
    if blur_interval > 0 and blur_std > 0:
        # only blur at certain iterations, as blurring is expensive
        if not iteration % blur_interval:
            # define standard deviations for blur kernel
            blur_kernel_std = [0, blur_std, blur_std, blur_std]
            
            # blur along height and width, but not along channel (color) dimension
            blur_kernel_std[ch_dim] = 0
            
            # perform blurring
            visualization = gaussian_filter(visualization, sigma=blur_kernel_std)

    # regularizer #3
    # apply value limit
    if value_percentile > 0:
        # find absolute values
        abs_visualization = abs(visualization)
        
        # find mask of high values (values above chosen value percentile)
        high_value_mask = abs_visualization >= np.percentile(abs_visualization, value_percentile)
        
        # apply to image to set pixels with small values to zero
        visualization *= high_value_mask

    # regularizer #4
    # apply norm limit
    if norm_percentile > 0:
        # compute pixel norms along channel (color) dimension
        pixel_norms = np.linalg.norm(visualization, axis=ch_dim)
        
        # find initial mask of high norms (norms above chosen norm percentile)
        high_norm_mask = pixel_norms >= np.percentile(pixel_norms, norm_percentile)
        
        # expand mask to account for colors
        high_norm_mask = expand_for_color(high_norm_mask, ch_dim)

        # apply to image to set pixels with small norms to zero
        visualization *= high_norm_mask

    # regularizer #5
    # apply contribution limit
    if contribution_percentile > 0:
        # predict the contribution of each pixel
        predicted_contribution = -visualization * pixel_gradients
    
        # sum over channel (color) dimension
        contribution = predicted_contribution.sum(ch_dim)

        # find initial mask of high contributions (contr. above chosen contr. percentile)
        high_contribution_mask = contribution >= np.percentile(contribution, contribution_percentile)

        # expand mask to account for colors
        high_contribution_mask = expand_for_color(high_contribution_mask, ch_dim)

        # apply to image to set pixels with small contributions to zero
        visualization *= high_contribution_mask

    # regularizer #6
    # apply absolute contribution limit
    if abs_contribution_percentile > 0:
    
        # alternative approach
        # predict the contribution of each pixel
        predicted_contribution = -visualization * pixel_gradients
    
        # sum over channel (color) dimension, and find absolute value
        abs_contribution = abs(predicted_contribution.sum(ch_dim))
        
        # find initial mask of high absolute contributions (abs. contr. above chosen abs. contr. percentile)
        high_abs_contribution_mask = abs_contribution >= np.percentile(abs_contribution, abs_contribution_percentile)

        # expand mask to account for colors
        high_abs_contribution_mask = expand_for_color(high_abs_contribution_mask, ch_dim)

        # apply to image to set pixels with small absolute contributions to zero
        visualization *= high_abs_contribution_mask

    return visualization


# TODO: use expand_dims instead of np.newaxis?
# use to expand a (batch, height, width)-numpy array with a channel (color) dimension
def expand_for_color(np_array, ch_dim):
    if ch_dim == 1:
        # for numpy arrays on form (batch, color, height, width)
        np_array = np.tile(np_array[:, np.newaxis, :, :], (1, 3, 1, 1))
    elif ch_dim == 3:
        # for numpy arrays on form (batch, height, width, color)
        np_array = np.tile(np_array[:, :, :, np.newaxis], (1, 1, 1, 3))
    else:
        raise ValueError('channel (color) dimension {} not recognized as legal dimension value'.format(ch_dim))
    return np_array


def main():
    # create model to generate gradients from
    model = create_model()

    # 130 flamingo, 351 hartebeest, 736 pool table, 850 teddy bear
    for class_index in [130, 351, 736, 850]:
        print('Processing class {}'.format(class_index))
        
        # used to time generation of each image
        start_time = time()
        
        # create and save gradient function for current class
        iterate = get_gradient_function(model.input, model.output, class_index)
    
        # create an initial visualization image, and locate its channel (color) dimension
        visualization, ch_dim = create_initial_image(model.input_shape)
        
        # TODO: delete discrimination when done with testing
        loss_value = 0.0
        if class_index == 736:
        # if class_index:
            # perform gradient ascent update with or without regularization for n steps
            for i in range(no_of_iterations):
                
                # compute loss and gradient values
                # input 0 for test phase
                loss_value, pixel_gradients = iterate([visualization, 0])
        
                # update visualization image
                visualization += pixel_gradients * learning_rate
        
                # if regularization has been activated, regularize image
                if regularize:
                    visualization = apply_ensemble_regularization(visualization, ch_dim, pixel_gradients, i)
        
                # print('Current loss value:', loss_value)
                print('Round {} finished.'.format(i))

            # process visualization to match with standard image dimensions
            visualization_image = deprocess(visualization, ch_dim)
            
            print('Class {} visualization completed in {}s'.format(class_index, time() - start_time))
        
            # save visualization image, complete with info about creation environment
            save_visualization(visualization_image, class_index, loss_value)

main()
