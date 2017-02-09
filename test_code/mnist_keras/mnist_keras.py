
from time import time
from os import mkdir, listdir
from os.path import join, dirname

import numpy as np

from keras.models import Model, load_model
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, Dropout, Flatten
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical

# needed to create custom callbacks
from keras.callbacks import Callback

# find path to save models
save_path = join(dirname(__file__), 'networks')


# saves accuracy at each finished training batch
class AccuracyListSaver(Callback):

    # NOTE: all imports in class could be performed globally

    # find results path
    from os.path import dirname, join
    results_path = join(dirname(__file__), 'results')

    model_no = None

    def __init__(self, model_no):
        super(AccuracyListSaver, self).__init__()
        self.model_no = model_no

    def on_train_begin(self, logs={}):
        # ensure file creation
        with open(self.results_path + '/batch_accuracy.txt', 'w') as f:
            f.write('')

    def on_batch_end(self, batch, logs={}):
        # write new accuracy line
        with open(self.results_path + '/batch_accuracy.txt', 'a') as f:
            f.write(str(logs['acc']) + '\n')


# saves loss at each finished training batch
class LossListSaver(Callback):

    # NOTE: all imports in class could be performed globally

    # find results path
    from os.path import dirname, join
    results_path = join(dirname(__file__), 'results')

    model_no = None

    def __init__(self, model_no):
        super(LossListSaver, self).__init__()
        self.model_no = model_no

    def on_train_begin(self, logs={}):
        # ensure file creation
        with open(self.results_path + '/batch_loss.txt', 'w') as f:
            f.write('')

    def on_batch_end(self, batch, logs={}):
        # write new loss line
        with open(self.results_path + '/batch_loss.txt', 'a') as f:
            f.write(str(logs['loss']) + '\n')


# saves activation arrays for each layer as tuples: (layer-name, array)
class ActivationTupleListSaver(Callback):

    # NOTE: all imports in class could be performed globally

    # find results path
    from os.path import dirname, join
    results_path = join(dirname(__file__), 'results')

    model_no = None
    input_tensor = None

    def __init__(self, model_no):
        import numpy as np

        super(ActivationTupleListSaver, self).__init__()
        self.model_no = model_no

        # get one random image from training data to use as input
        training_data, _, _, _ = load_data()
        self.input_tensor = training_data[np.random.randint(len(training_data))].reshape(1, 28, 28, 1)

    def on_epoch_end(self, batch, logs={}):
        import keras.backend as K
        import pickle

        # initialize layer tuple list with image
        layer_tuples = []

        # for all layers, get and save activation tensor
        for layer in self.model.layers:
            # create function using keras-backend for getting activation tensor
            get_activation_tensor = K.function([self.model.input, K.learning_phase()], [layer.output])

            # save tuple (layer name, layer's activation tensor)
            # NOTE: learning phase 0 is testing and 1 is training (difference unknown as this point)
            layer_tuples.append((layer.name, get_activation_tensor([self.input_tensor, 0])[0]))

        with open(self.results_path + '/layer_dict_%d.pickle' % self.model_no, 'wb') as f:
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

    # create and compile model with chosen hyperparameters
    model = Model(input=inputs, output=predictions)
    model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])

    # save model
    model_no = None
    if with_save:
        model_no = save_to_disk(model)

    print('\nModel successfully created')

    return model, model_no


def train(model, model_no, no_of_epochs=10):

    # train top layers of model (self-defined layers)
    print('\n\nCommence MNIST model training\n')

    # initialize custom callbacks
    custom_callbacks = [AccuracyListSaver(model_no), LossListSaver(model_no), ActivationTupleListSaver(model_no)]
    
    # get data
    training_data, training_targets, test_data, test_targets = load_data()
    
    # train with chosen hyperparameters
    model.fit(training_data, training_targets, nb_epoch=no_of_epochs, batch_size=128, shuffle=True,
              verbose=1, callbacks=custom_callbacks, validation_data=(test_data, test_targets))

    # save after training
    save_to_disk(model, model_no)

    print('\nCompleted MNIST model training\n')

    return model


def test(model, no_of_tests=1, verbose=True):

    # get validation data
    _, _, test_data, test_targets = load_data()

    # find indices of random images to test on
    random_indices = np.random.randint(len(test_data), size=no_of_tests)

    # test for all indices and count correctly classified
    correctly_classified = 0
    for i in random_indices:

        # get model classification
        classification = model.predict(test_data[i].reshape(1, 28, 28, 1))

        # find correct classification
        correct = test_targets[i]

        # count correctly classified, and print if incorrect
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

    # make path, if not exists
    try:
        mkdir(save_path)
        print('networks-folder created')
    except FileExistsError:
        # file exists, which is want we want
        pass

    # find model number, if not specified
    if model_no is None:
        model_no = len(listdir(save_path))

    model.save('%s/mnist_model_%d.h5' % (save_path, model_no))

    print('\nModel saved as mnist_model_%d.h5' % model_no)

    return model_no


def load_from_disk(model_no):
    
    # load chosen model number
    model = load_model('%s/mnist_model_%d.h5' % (save_path, model_no))

    print('\nModel loaded from mnist_model_%d.h5' % model_no)

    return model


def main():

    # create a model, then train and test it.

    start_time = time()

    model, model_no = create_model()

    model = train(model, model_no)

    test(model, 1000, True)

    print('This took %.2f seconds' % (time() - start_time))

main()
