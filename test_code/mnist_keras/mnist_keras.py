import pickle
from time import time
from os import mkdir, listdir
from os.path import join, dirname

import numpy as np

from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, Dropout, Flatten
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from keras.callbacks import Callback
from keras.preprocessing import image


file_path = dirname(__file__)
save_path = join(file_path, 'model_weights')
result_path = join(file_path, 'results')


# TODO: maybe save acc and loss as pickle-files instead of text?


class AccuracyListSaver(Callback):

    model_no = None

    def __init__(self, model_no):
        super(AccuracyListSaver, self).__init__()
        self.model_no = model_no
        create_result_folder()

    def on_train_begin(self, logs={}):
        # ensure file creation
        with open(result_path + '/batch_accuracy_%d.txt' % self.model_no, 'w') as f:
            f.write('')

    def on_batch_end(self, batch, logs={}):
        # write new accuracy line
        with open(result_path + '/batch_accuracy_%d.txt' % self.model_no, 'a') as f:
            f.write(str(logs['acc']) + '\n')


class LossListSaver(Callback):

    model_no = None

    def __init__(self, model_no):
        super(LossListSaver, self).__init__()
        self.model_no = model_no
        create_result_folder()

    def on_train_begin(self, logs={}):
        # ensure file creation
        with open(result_path + '/batch_loss_%d.txt' % self.model_no, 'w') as f:
            f.write('')

    def on_batch_end(self, batch, logs={}):
        # write new loss line
        with open(result_path + '/batch_loss_%d.txt' % self.model_no, 'a') as f:
            f.write(str(logs['loss']) + '\n')


class ActivationTupleListSaver(Callback):

    model_no = None
    input_tensor = None

    def __init__(self, model_no):
        super(ActivationTupleListSaver, self).__init__()
        self.model_no = model_no

        # get one random image from training data to use as input
        training_data, _, _, _ = load_data()
        self.input_tensor = training_data[np.random.randint(len(training_data))].reshape(1, 28, 28, 1)

        create_result_folder()

    def on_epoch_end(self, batch, logs={}):

        # initialize layer tuple list with image
        layer_tuples = []

        # for all layers, get and save activation tensor
        for layer in self.model.layers:
            # create function using keras-backend for getting activation tensor
            get_activation_tensor = K.function([self.model.input, K.learning_phase()], [layer.output])

            # save tuple (layer name, layer's activation tensor)
            # NOTE: learning phase 0 is testing and 1 is training (difference unknown as this point)
            layer_tuples.append((layer.name, get_activation_tensor([self.input_tensor, 0])[0]))

        with open(result_path + '/layer_dict_%d.pickle' % self.model_no, 'wb') as f:
            pickle.dump(layer_tuples, f)


def create_model(with_save=True):

    # define model
    inputs = Input(shape=(28, 28, 1))
    x = Convolution2D(32, 3, 3, activation='relu', border_mode='valid')(inputs)
    x = Convolution2D(32, 3, 3, activation='relu', border_mode='valid')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    predictions = Dense(10, activation='softmax')(x)

    model = Model(input=inputs, output=predictions)
    model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])

    model_no = None
    if with_save:
        model_no = save_to_disk(model)

    print('\nModel successfully created')

    return model, model_no


def train(model, model_no, no_of_epochs=10):

    # train top layers of model (self-defined layers)
    print('\n\nCommence MNIST model training\n')

    acc_saver = AccuracyListSaver(model_no)
    loss_saver = LossListSaver(model_no)
    layer_saver = ActivationTupleListSaver(model_no)
    training_data, training_targets, test_data, test_targets = load_data()
    model.fit(training_data, training_targets, nb_epoch=no_of_epochs, batch_size=128, shuffle=True,
              verbose=1, callbacks=[acc_saver, loss_saver, layer_saver], validation_data=(test_data, test_targets))

    save_to_disk(model, model_no)

    print('\nCompleted MNIST model training\n')

    return model


def test(model, no_of_tests=1, verbose=True):

    _, _, test_data, test_targets = load_data()

    random_indices = np.random.randint(len(test_data), size=no_of_tests)

    correctly_classified = 0
    for i in random_indices:

        classification = model.predict(test_data[i].reshape(1, 28, 28, 1))

        correct = test_targets[i]

        if np.argmax(classification) == np.argmax(correct):
            correctly_classified += 1
        elif verbose:
            print('Incorrectly classified %d as %d' % (np.argmax(correct), np.argmax(classification)))

    print('Model correctly classified %d/%d MNIST images' % (correctly_classified, no_of_tests))


def load_data():

    # load
    (training_data, training_targets), (test_data, test_targets) = mnist.load_data()

    # reshape data
    training_data = training_data.reshape(training_data.shape[0], 28, 28, 1)
    test_data = test_data.reshape(test_data.shape[0], 28, 28, 1)

    # normalize data
    training_data = training_data.astype('float32')
    test_data = test_data.astype('float32')
    training_data /= 255.0
    test_data /= 255.0

    # make targets categorical
    training_targets = to_categorical(training_targets, 10)
    test_targets = to_categorical(test_targets, 10)

    return training_data, training_targets, test_data, test_targets


def save_to_disk(model, model_no=None):

    try:
        mkdir(save_path)
        print('model_weights folder created')
    except FileExistsError:
        # file exists, which is want we want
        pass

    if model_no is None:
        model_no = len(listdir(save_path))

    model.save_weights(save_path + '/mnist_weights_%d.h5' % model_no)

    print('\nModel weights saved as mnist_weights_%d.h5' % model_no)

    return model_no


def load_from_disk(model_no):

    model, _ = create_model(with_save=False)

    model.load_weights('./model_weights/mnist_weights_%d.h5' % model_no)

    print('\nModel created with loaded weigths from mnist_weights_%d.h5' % model_no)

    return model


def create_result_folder():
        try:
            mkdir(result_path)
        except FileExistsError:
            # file exists, which we wanted
            pass


def main():

    start_time = time()

    model, model_no = create_model()

    model = train(model, model_no)

    # test(model, 1000, True)

    print('This took %.2f seconds' % (time() - start_time))

main()
