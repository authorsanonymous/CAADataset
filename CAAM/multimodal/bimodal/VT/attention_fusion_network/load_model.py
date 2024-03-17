import numpy as np
import keras

from keras.models import Model, load_model
from keras.layers import Dense, Input, Concatenate, Dropout, Add, Lambda


def load_model(location=None):

	if(location != None):
		model = keras.models.load_model(location)
		print("Loaded the model.")
		return model

	facial_X_pose = Input(shape = (1542,))
	gaze_X_action = Input(shape = (1040,))
	transcript = Input(shape = (200,))

	X_gender = Input(shape = (1,))


	facial_shortened = Dense(600, activation = 'relu')(facial_X_pose)
	facial_shortened = Dense(450, activation = 'relu')(facial_shortened)

	gaze_X_action_shortened = Dense(570, activation = 'relu')(gaze_X_action)
	gaze_X_action_shortened = Dense(450, activation = 'relu')(gaze_X_action_shortened)

	transcript_elongated = Dense(315, activation = 'relu')(transcript)
	transcript_elongated = Dense(450, activation = 'relu')(transcript_elongated)


	B = Concatenate(axis = 1)([facial_shortened, gaze_X_action_shortened, transcript_elongated])


	P = Dense(300, activation = 'tanh')(B)

	alpha = Dense(3, activation = 'softmax')(P)

	F = Lambda(lambda x : alpha[:,0:1]*facial_shortened + alpha[:,1:2]*gaze_X_action_shortened + alpha[:,2:3]*transcript_elongated)(alpha)

	Y = Concatenate(axis = -1)([F, X_gender])

	Y = Dense(310, activation = 'relu')(Y)
	Y = Dropout(rate = 0.5)(Y)
	
	Y = Dense(83, activation = 'relu')(Y)
	Y = Dropout(rate = 0.4)(Y)

	Y = Dense(1, activation = None)(Y)

	model = Model(inputs = [facial_X_pose, gaze_X_action, transcript, X_gender], outputs = Y)

	print("Created a new model.")

	return model



if(__name__ == "__main__"):
	m = load_model()