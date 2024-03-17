import numpy as np
import keras

from keras.models import Model, load_model
from keras.layers import Dense, Input, Concatenate, Dropout, Add, Lambda



def load_model(location=None):

	if(location != None):
		model = keras.models.load_model(location)
		print("Loaded the model.")
		return model

	FORMANT = Input(shape = (418,))
	facial = Input(shape = (1542,))
	transcript = Input(shape = (200,))

	X_gender = Input(shape = (1,))


	FORMANT_shortened = Dense(450, activation = 'relu')(FORMANT)

	facial_shortened = Dense(600, activation = 'relu')(facial)
	facial_shortened = Dense(450, activation = 'relu')(facial_shortened)


	transcript_elongated = Dense(315, activation = 'relu')(transcript)
	transcript_elongated = Dense(450, activation = 'relu')(transcript_elongated)


	B = Concatenate(axis = 1)([FORMANT_shortened, facial_shortened, transcript_elongated])

	P = Dense(300, activation = 'tanh')(B)

	alpha = Dense(4, activation = 'softmax')(P)

	F = Lambda(lambda x : alpha[:,0:1]*FORMANT_shortened + alpha[:,1:2]*facial + alpha[:,2:3]*transcript_elongated)(alpha)

	Y = Concatenate(axis = -1)([F, X_gender])

	Y = Dense(210, activation = 'relu')(Y)
	Y = Dropout(rate = 0.5)(Y)
	
	Y = Dense(83, activation = 'relu')(Y)
	Y = Dropout(rate = 0.3)(Y)
	#
	Y = Dense(23, activation='relu')(Y)
	Y = Dropout(rate=0.3)(Y)

	Y = Dense(1, activation = None)(Y)

	model = Model(inputs = [FORMANT,facial, X_gender, transcript], outputs = Y)

	print("Created a new model.")

	return model



if(__name__ == "__main__"):
	m = load_model()