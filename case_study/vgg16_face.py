from os.path import dirname

from keras_vggface.vggface import VGGFace

from keras.models import Model
from keras.layers import Input, Dense, Concatenate, Dropout
from keras.optimizers import Adam

from case_study.helpers import train_model

# find path to save networks and results
save_path = dirname(__file__)

# set model parameters
# choose if model is to be experimental (False & False = baseline)
extra_input = False
extra_output = True
assert not (extra_input and extra_output)
# amount of classification possibilities
id_amount = 98
# expression vector size
expression_amount = 7

# set training parameters
no_of_epochs = 50
batch_size = 512


"""

IMPORTANT NOTE:

In VGGFace, the top layers have activations in its own layers, therefore input and output tensors of last layer
should be accessed in the following manner (see numbers):

input_tensor = model.layers[-2].input
output_tensor = model.layers[-1].output

"""


# create model only consisting of top layers using features from a VGGFace base model
def create_top_model():

	# load VGGFace model
	vgg_face_model = VGGFace(include_top=True, input_shape=(224, 224, 3))

	# get output shape of VGGFace base model (excluding last layer, see note)
	base_output_shape = vgg_face_model.layers[-2].input_shape[1:]

	# define model input
	model_input = Input(shape=base_output_shape, name='feat_input')

	if extra_input:
		# define extra input
		expression_input = Input(shape=(expression_amount,), name='expression_input')

		# define experimental structure
		# use Concatenate layer to merge expression input with standard input
		x = model_input
		x = Dropout(0.5)(x)
		x = Dense(1024, activation='relu')(x)
		x = Dropout(0.5)(x)
		x = Concatenate()([x, expression_input])
		x = Dense(1024, activation='relu')(x)
		x = Dropout(0.5)(x)
		id_output = Dense(id_amount, activation='softmax', name='id_output')(x)
		model_output = id_output

		# redefine model input to receive two inputs
		model_input = [model_input, expression_input]

	elif extra_output:

		# define experimental structure
		x = model_input
		x = Dropout(0.5)(x)
		x = Dense(1024, activation='relu')(x)
		x = Dropout(0.5)(x)
		x = Dense(1024, activation='relu')(x)
		x_id = Dropout(0.5)(x)
		id_output = Dense(id_amount, activation='softmax', name='id_output')(x_id)

		# define extra output (with extra layers)
		x = x_id
		x = Dense(1024, activation='relu', name='fc_exp3')(x)
		x = Dropout(0.5)(x)
		expression_output = Dense(expression_amount, activation='softmax', name='expression_output')(x)

		# define model output to yield two outputs
		model_output = [id_output, expression_output]

	else:
		# define baseline structure
		x = model_input
		x = Dropout(0.5)(x)
		x = Dense(1024, activation='relu')(x)
		x = Dropout(0.5)(x)
		x = Dense(1024, activation='relu')(x)
		x = Dropout(0.5)(x)
		id_output = Dense(id_amount, activation='softmax', name='id_output')(x)
		model_output = id_output

	# create and compile model
	custom_vgg_model = Model(model_input, model_output)
	custom_vgg_model.compile(optimizer=Adam(lr=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])

	return custom_vgg_model


def main():

	model_type = 'baseline'
	if extra_input:
		model_type = 'extra_input'
	elif extra_output:
		model_type = 'extra_output'

	model = create_top_model()
	train_model(model, no_of_epochs, batch_size, model_type, save_path)


main()
