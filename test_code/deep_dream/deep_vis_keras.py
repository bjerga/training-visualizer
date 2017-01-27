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

# VGG16 mean values
MEAN_VALUES = np.array([103.939, 116.779, 123.68])

# Update coefficient
learning_rate = 4000.

# how many times we update image
no_of_iterations = 500

# specify L2-decay
# used to prevent a small number of extreme pixel values from dominating the output image
# l2_decay = 0.0001
l2_decay = 0.0

# specify frequency of blurring and standard deviation for kernel for Gaussian blur
# used to penalize high frequency information in the output image
blur_every = 4
# standard deviation values between 0.0 and 0.3 work poorly, according to yosinski
# blur_std = 1.0
blur_std = 0.5

# specify value percentile limit
# used to induce sparsity by setting pixels with small absolute value to 0
value_percentile = 0

# specify norm percentile limit
# used to induce sparsity by setting pixels with small norm to 0
norm_percentile = 50

# specify contribution percentile limit
# used to induce sparsity by setting pixels with small contribution to 0
contribution_percentile = 0

# specify absolute contribution percentile limit
# used to induce sparsity by setting pixels with small absolute contribution to 0
abs_contribution_percentile = 0

# choose whether to include regularization
regularize = True


# util function to convert a tensor into a valid image
def deprocess(vis_tensor, color_axis):

    # remove batch dimension, and alter color axis accordingly
    img = vis_tensor[0]
    color_axis -= 1

    # create shape with correct color dimension
    new_shape = [1, 1, 1]
    new_shape[color_axis] = 3

    # add VGG16 mean values
    img += MEAN_VALUES.reshape(new_shape)

    if color_axis == 0:
        # alter dimensions from (color, height, width) to (height, width, color)
        img = img.transpose((1, 2, 0))

    img = np.clip(img, 0, 255).astype('uint8')  # clip in [0;255] and convert to int
    
    return img


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


def save_visualization(model_input_shape, kept_images):
    
    # define output path and make folder
    output_path = join(dirname(__file__), 'output')
    try:
        mkdir(output_path)
    except FileExistsError:
        # folder exists, which is what we wanted
        pass

    # Compute the size of the grid
    n = int(np.ceil(np.sqrt(len(kept_images))))
    
    # build a black picture with enough space for the kept_images
    # img_height = model.input_shape[2]
    # img_width = model.input_shape[3]
    img_height = model_input_shape[1]
    img_width = model_input_shape[2]
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
    
    # save the resulting image to disk
    if regularize:
        image_name = 'regularized_%d' % time()
    else:
        image_name = 'vanilla_%d' % time()

    scipy.misc.toimage(stitched_res, cmin=0, cmax=255).save(join(output_path, image_name + '.png'))
    # Do not use scipy.misc.imsave because it will normalize the image pixel value between 0 and 255

    # also save a txt-file containing information about creation environment
    with open(join(output_path, image_name + '_info.txt'), 'a') as f:
        f.write('Learning rate: %f\n' % learning_rate)
        f.write('Number of iterations: %d\n' % no_of_iterations)
        if regularize:
            f.write('L2-decay: %f\n' % l2_decay)
            f.write('Blur every and std: %d & %f\n' % (blur_every, blur_std))
            f.write('Value percentile: %d\n' % value_percentile)
            f.write('Norm percentile: %d\n' % norm_percentile)
            f.write('Contribution percentile: %d\n' % contribution_percentile)
            f.write('Abs. contribution percentile: %d\n' % abs_contribution_percentile)


def get_gradient_function(model_input, model_output, class_index):
    
    # The loss is the activation of the neuron for the chosen class
    loss = model_output[0, class_index]
    
    # we compute the gradient of the input picture w.r.t. this loss
    grads = K.gradients(loss, model_input)[0]
    
    # this function returns the loss and grads given the input picture
    # also add a flag to disable the learning phase (in our case dropout)
    return K.function([model_input, K.learning_phase()], [loss, grads])


def create_initial_image(model_input_shape):
    
    np.random.seed(1337)  # for reproducibility
    # we start from a gray image with some random noise
    
    # return a random, initial image, and the color axis of the image
    # add (1,) for batch axis
    return np.random.normal(0, 10, (1,) + model_input_shape[1:]), model_input_shape.index(3)


def ensemble_regularization(input_img_data, grads_value, iteration, color_axis):
    
    # regularizer #1
    # apply L2-decay
    input_img_data *= (1 - l2_decay)

    # regularizer #2
    # apply Gaussian blur
    if blur_every is not 0 and blur_std > 0:
        if not iteration % blur_every:
            # define standard deviations for blur kernel
            blur_kernel_std = [0, blur_std, blur_std, blur_std]
            
            # blur along height and width, but not channels (colors)
            blur_kernel_std[color_axis] = 0
            
            # perform blurring
            input_img_data = gaussian_filter(input_img_data, sigma=blur_kernel_std)

    # regularizer #3
    # apply value limit
    if value_percentile > 0:
        
        # alternative approach
        # get absolute values
        abs_input_img_data = abs(input_img_data)
        
        # find mask of high values (values above chosen value percentile)
        high_value_mask = abs_input_img_data >= np.percentile(abs_input_img_data, value_percentile)
        
        # apply to image remove small values
        # input_img_data *= high_value_mask
        alt_img = input_img_data * high_value_mask
        
        # original approach
        small_entries = (abs(input_img_data) < np.percentile(abs(input_img_data), value_percentile))
        input_img_data -= input_img_data * small_entries  # e.g. set smallest 50% of input_img_data to zero

        for i in range(len(input_img_data[0])):
            for j in range(len(input_img_data[0, i])):
                for k in range(len(input_img_data[0, i, j])):
                    if input_img_data[0, i, j, k] != alt_img[0, i, j, k]:
                        print('\n\nVALUE NOT EQUAL\n\n')
        print('All is well with value')

    # regularizer #4
    # apply norm limit
    if norm_percentile > 0:

        # alternative approach
        # compute pixel norms along channel (color) axis
        pixel_norms = np.linalg.norm(input_img_data, axis=color_axis)
        
        # find initial mask of high norms (norms above chosen norm percentile)
        high_norm_mask = pixel_norms >= np.percentile(pixel_norms, norm_percentile)
        
        # expand mask to account for channels (colors)
        high_norm_mask = expand_for_color(high_norm_mask, color_axis)
        
        # input_img_data *= high_norm_mask
        alt_img = input_img_data * high_norm_mask

        # original approach
        # pxnorms = np.linalg.norm(input_img_data, axis=1)
        pxnorms = np.linalg.norm(input_img_data, axis=3)
        smallpx = pxnorms < np.percentile(pxnorms, norm_percentile)
        # smallpx3 = np.tile(smallpx[:, np.newaxis, :, :], (1, 3, 1, 1))
        smallpx3 = np.tile(smallpx[:, :, :, np.newaxis], (1, 1, 1, 3))
        input_img_data -= input_img_data * smallpx3

        for i in range(len(input_img_data[0])):
            for j in range(len(input_img_data[0, i])):
                for k in range(len(input_img_data[0, i, j])):
                    if input_img_data[0, i, j, k] != alt_img[0, i, j, k]:
                        print('\n\nNORM NOT EQUAL\n\n')
        print('All is well with norm')

    # regularizer #5
    # apply contribution limit
    if contribution_percentile > 0:
    
        # alternative approach
        pred_0_contribution = -input_img_data * grads_value
    
        # sum over color channels
        contribution = pred_0_contribution.sum(color_axis)
        high_contribution_mask = contribution >= np.percentile(contribution, contribution_percentile)

        # expand mask to account for channels (colors)
        high_contribution_mask = expand_for_color(high_contribution_mask, color_axis)
        
        # input_img_data *= high_contribution_mask
        alt_img = input_img_data * high_contribution_mask
        
        # original approach
        pred_0_contribution = grads_value * -input_img_data
        # contribution = pred_0_contribution.sum(1)  # sum over color channels
        contribution = pred_0_contribution.sum(3)  # sum over color channels
        smallben = contribution < np.percentile(contribution, contribution_percentile)
        # smallben3 = np.tile(smallben[:, np.newaxis, :, :], (1, 3, 1, 1))
        smallben3 = np.tile(smallben[:, :, :, np.newaxis], (1, 1, 1, 3))
        input_img_data -= input_img_data * smallben3

        for i in range(len(input_img_data[0])):
            for j in range(len(input_img_data[0, i])):
                for k in range(len(input_img_data[0, i, j])):
                    if input_img_data[0, i, j, k] != alt_img[0, i, j, k]:
                        print('\n\nCONTR. NOT EQUAL\n\n')
        print('All is well with contr.')

    # regularizer #6
    # apply absolute contribution limit
    if abs_contribution_percentile > 0:
    
        # alternative approach
        pred_0_contribution = -input_img_data * grads_value
    
        # sum over color channels
        abs_contribution = abs(pred_0_contribution.sum(color_axis))
        high_abs_contribution_mask = abs_contribution >= np.percentile(abs_contribution, abs_contribution_percentile)

        # expand mask to account for channels (colors)
        high_abs_contribution_mask = expand_for_color(high_abs_contribution_mask, color_axis)
        
        # input_img_data *= high_abs_contribution_mask
        alt_img = input_img_data * high_abs_contribution_mask

        # original approach
        pred_0_contribution = grads_value * -input_img_data
        # contribution = pred_0_contribution.sum(1)  # sum over color channels
        contribution = pred_0_contribution.sum(3)  # sum over color channels
        smallaben = abs(contribution) < np.percentile(abs(contribution), abs_contribution_percentile)
        # smallaben3 = np.tile(smallaben[:, np.newaxis, :, :], (1, 3, 1, 1))
        smallaben3 = np.tile(smallaben[:, :, :, np.newaxis], (1, 1, 1, 3))
        input_img_data -= input_img_data * smallaben3

        for i in range(len(input_img_data[0])):
            for j in range(len(input_img_data[0, i])):
                for k in range(len(input_img_data[0, i, j])):
                    if input_img_data[0, i, j, k] != alt_img[0, i, j, k]:
                        print('\n\nABS CONTR. NOT EQUAL\n\n')
        print('All is well with abs contr.')
        
    return input_img_data


# use to expand a [batch, height, width]-numpy array with a color dimension
def expand_for_color(np_array, color_axis):
    if color_axis == 1:
        # for numpy arrays on form [batch, color, height, width]
        np_array = np.tile(np_array[:, np.newaxis, :, :], (1, 3, 1, 1))
    elif color_axis == 3:
        # for numpy arrays on form [batch, height, width, color]
        np_array = np.tile(np_array[:, :, :, np.newaxis], (1, 1, 1, 3))
    else:
        raise ValueError('Color axis %d not recognized as legal axis' % color_axis)
    return np_array


def main():
    # create model to generate gradients from
    model = create_model()
    
    # list of images generated from regularized gradient ascent
    generated_images = []

    # 130 flamingo, 351 hartebeest, 736 pool table, 850 teddy bear
    for class_index in [130, 351, 736, 850]:
        print('Processing class %d' % class_index)
        
        # used to time generation of each image
        start_time = time()
        
        # create and save gradient function for current class
        iterate = get_gradient_function(model.input, model.output, class_index)
    
        # get initial image and its color axis
        input_img_data, color_axis = create_initial_image(model.input_shape)
        
        loss_value = None
        if class_index == 736:
        # if class_index:
            # run gradient ascent for n steps
            for i in range(no_of_iterations):
                
                # input 0 for test phase
                loss_value, grads_value = iterate([input_img_data, 0])
        
                # Apply gradient to image
                input_img_data += grads_value * learning_rate
        
                if regularize:
                    input_img_data = ensemble_regularization(input_img_data, grads_value, i, color_axis)
        
                # print('Current loss value:', loss_value)
                print('Round %d finished.' % i)

        # decode the resulting input image and add it to the list
        img = deprocess(input_img_data, color_axis)
        generated_images.append((img, loss_value))
        print('Filter %d processed in %ds' % (class_index, time() - start_time))
    
    save_visualization(model.input_shape, generated_images)

main()
