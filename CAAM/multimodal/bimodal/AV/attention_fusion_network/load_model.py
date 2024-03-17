import numpy as np
import keras
from keras import Model
from keras.layers import Dense, Concatenate, Lambda, Dropout, Input
from keras import backend as K


def load_model(location=None):

	if(location != None):
		model = keras.models.load_model(location)
		print("Loaded the model.")
		return model

	FORMANT = Input(shape = (418,))
	facial = Input(shape = (1542,))

	X_gender = Input(shape = (1,))


	FORMANT_shortened = Dense(450, activation = 'relu')(FORMANT)

	facial_shortened = Dense(600, activation = 'relu')(facial)
	facial_shortened = Dense(450, activation = 'relu')(facial_shortened)



	B = Concatenate(axis = 1)([FORMANT_shortened, facial_shortened])


	P = Dense(200, activation = 'tanh')(B)

	alpha = Dense(3, activation = 'softmax')(P)

	F = Lambda(lambda x : alpha[:,0:1]*FORMANT_shortened + alpha[:,1:2]*facial_shortened)

	Y = Concatenate(axis = -1)([F, X_gender])

	Y = Dense(210, activation = 'relu')(Y)
	Y = Dropout(rate = 0.25)(Y)
	
	Y = Dense(63, activation = 'relu')(Y)
	Y = Dropout(rate = 0.2)(Y)

	Y = Dense(1, activation = None)(Y)

	model = Model(inputs = [FORMANT, facial, X_gender], outputs = Y)

	print("Created a new model.")

	return model



if(__name__ == "__main__"):
	m = load_model()