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

# VGG16 mean values
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
blur_every = 4
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

# utility function used to convert a tensor into a savable image
def deprocess(vis_tensor, color_axis):

    # remove batch dimension, and alter color axis accordingly
    img = vis_tensor[0]
    color_axis -= 1

    # create shape with correct color dimension
    new_shape = [1, 1, 1]
    new_shape[color_axis] = 3

    # add VGG16 mean values
    img += VGG16_MEAN_VALUES.reshape(new_shape)

    if color_axis == 0:
        # alter dimensions from (color, height, width) to (height, width, color)
        img = img.transpose((1, 2, 0))

    img = np.clip(img, 0, 255).astype('uint8')  # clip in [0;255] and convert to int
    
    return img


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
        image_name = 'regularized'
    else:
        image_name = 'vanilla'
    image_name += '_{}_{}'.format(class_index, time())

    # save the resulting image to disk
    scipy.misc.toimage(img, cmin=0, cmax=255).save(join(output_path, image_name + '.png'))
    # avoid scipy.misc.imsave because it will normalize the image pixel value between 0 and 255

    # also save a txt-file containing information about creation environment and obtained loss
    with open(join(output_path, image_name + '_info.txt'), 'a') as f:
        f.write('Learning rate: {}\n'.format(learning_rate))
        f.write('Number of iterations: {}\n'.format(no_of_iterations))
        f.write('----------\n')
        if regularize:
            f.write('L2-decay: {}\n'.format(l2_decay))
            f.write('Blur every and std: {} & {}\n'.format(blur_every, blur_std))
            f.write('Value percentile: {}\n'.format(value_percentile))
            f.write('Norm percentile: {}\n'.format(norm_percentile))
            f.write('Contribution percentile: {}\n'.format(contribution_percentile))
            f.write('Abs. contribution percentile: {}\n'.format(abs_contribution_percentile))
        f.write('----------\n')
        f.write('Obtained loss value: {}\n'.format(loss_value))
        
    print('\nImage of class {} have been saved as {}.png\n'.format(class_index, image_name))


# returns the function to easily compute the input image gradients w.r.t. the activations
def get_gradient_function(model_input, model_output, class_index):
    
    # loss is the activation of the neuron for the chosen class
    loss = model_output[0, class_index]
    
    # gradients are computed from the visualization w.r.t. this loss
    grads = K.gradients(loss, model_input)[0]
    
    # return function returning the loss and grads given a visualization image
    # add a flag to disable the learning phase (e.g. when using dropout)
    return K.function([model_input, K.learning_phase()], [loss, grads])


# creates an initial image to manipulate into a visualization
def create_initial_image(model_input_shape):
    
    # set random seed to be able to reproduce initial state of image
    # used in testing only, and should be remove upon implementation with tool
    np.random.seed(1337)
    
    # return a random, initial image, and the color axis of the image
    # add (1,) for batch axis
    return np.random.normal(0, 10, (1,) + model_input_shape[1:]), model_input_shape.index(3)


# regularizes input image with various techniques
# each technique is activated by non-zero values for their respective global variables
def ensemble_regularization(deep_vis, color_axis, pixel_gradients, iteration):
    
    # regularizer #1
    # apply L2-decay
    if l2_decay > 0:
        deep_vis *= (1 - l2_decay)

    # regularizer #2
    # apply Gaussian blur
    if blur_every > 0 and blur_std > 0:
        # only blur at certain iterations, as blurring is expensive
        if not iteration % blur_every:
            # define standard deviations for blur kernel
            blur_kernel_std = [0, blur_std, blur_std, blur_std]
            
            # blur along height and width, but not along color axis
            blur_kernel_std[color_axis] = 0
            
            # perform blurring
            deep_vis = gaussian_filter(deep_vis, sigma=blur_kernel_std)

    # regularizer #3
    # apply value limit
    if value_percentile > 0:
        # find absolute values
        abs_deep_vis = abs(deep_vis)
        
        # find mask of high values (values above chosen value percentile)
        high_value_mask = abs_deep_vis >= np.percentile(abs_deep_vis, value_percentile)
        
        # apply to image to set pixels with small values to zero
        deep_vis *= high_value_mask

    # regularizer #4
    # apply norm limit
    if norm_percentile > 0:
        # compute pixel norms along color axis
        pixel_norms = np.linalg.norm(deep_vis, axis=color_axis)
        
        # find initial mask of high norms (norms above chosen norm percentile)
        high_norm_mask = pixel_norms >= np.percentile(pixel_norms, norm_percentile)
        
        # expand mask to account for colors
        high_norm_mask = expand_for_color(high_norm_mask, color_axis)

        # apply to image to set pixels with small norms to zero
        deep_vis *= high_norm_mask

    # regularizer #5
    # apply contribution limit
    if contribution_percentile > 0:
        # predict the contribution of each pixel
        predicted_contribution = -deep_vis * pixel_gradients
    
        # sum over color axis
        contribution = predicted_contribution.sum(color_axis)

        # find initial mask of high contributions (contr. above chosen contr. percentile)
        high_contribution_mask = contribution >= np.percentile(contribution, contribution_percentile)

        # expand mask to account for colors
        high_contribution_mask = expand_for_color(high_contribution_mask, color_axis)

        # apply to image to set pixels with small contributions to zero
        deep_vis *= high_contribution_mask

    # regularizer #6
    # apply absolute contribution limit
    if abs_contribution_percentile > 0:
    
        # alternative approach
        # predict the contribution of each pixel
        predicted_contribution = -deep_vis * pixel_gradients
    
        # sum over color axis, and find absolute value
        abs_contribution = abs(predicted_contribution.sum(color_axis))
        
        # find initial mask of high absolute contributions (abs. contr. above chosen abs. contr. percentile)
        high_abs_contribution_mask = abs_contribution >= np.percentile(abs_contribution, abs_contribution_percentile)

        # expand mask to account for colors
        high_abs_contribution_mask = expand_for_color(high_abs_contribution_mask, color_axis)

        # apply to image to set pixels with small absolute contributions to zero
        deep_vis *= high_abs_contribution_mask

    return deep_vis


# use to expand a [batch, height, width]-numpy array with a color dimension
def expand_for_color(np_array, color_axis):
    if color_axis == 1:
        # for numpy arrays on form [batch, color, height, width]
        np_array = np.tile(np_array[:, np.newaxis, :, :], (1, 3, 1, 1))
    elif color_axis == 3:
        # for numpy arrays on form [batch, height, width, color]
        np_array = np.tile(np_array[:, :, :, np.newaxis], (1, 1, 1, 3))
    else:
        raise ValueError('Color axis {} not recognized as legal axis value'.format(color_axis))
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
    
        # create an initial visualization image, and locate its color axis
        deep_vis, color_axis = create_initial_image(model.input_shape)
        
        # TODO: delete discrimination when done with testing
        loss_value = 0.0
        if class_index == 736:
        # if class_index:
            # perform gradient ascent update with or without regularization for n steps
            for i in range(no_of_iterations):
                
                # compute loss and gradient values
                # input 0 for test phase
                loss_value, pixel_gradients = iterate([deep_vis, 0])
        
                # update visualization image
                deep_vis += pixel_gradients * learning_rate
        
                # if regularization has been activated, regularize image
                if regularize:
                    deep_vis = ensemble_regularization(deep_vis, color_axis, pixel_gradients, i)
        
                # print('Current loss value:', loss_value)
                print('Round {} finished.'.format(i))

            # process visualization to match with standard image dimensions
            visualization_image = deprocess(deep_vis, color_axis)
            
            print('Class {} visualization completed in {}s'.format(class_index, time() - start_time))
        
            # save visualization image, complete with info about creation environment
            save_visualization(visualization_image, class_index, loss_value)

main()
