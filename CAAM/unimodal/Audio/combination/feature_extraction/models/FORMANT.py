import numpy as np
import keras

from keras.models import Model, Sequential, load_model
from keras.layers import Dense, CuDNNLSTM, Input, Concatenate, Dropout,CuDNNGRU
from keras import regularizers


def load_model(location = None):

	if(location != None):
		model = keras.models.load_model(location)
		print("Loaded the model.")
		return model

	X = Input(shape = (20000, 5,))
	X_gender = Input(shape = (1,))

	Y =CuDNNGRU(10, name = 'lstm_cell')(X)
	Y = Dropout(rate = 0.25)(Y)

	Y = Concatenate(axis = -1)([Y, X_gender])

	Y = Dense(6, activation = 'relu')(Y)
	Y = Dropout(rate = 0.25)(Y)

	Y = Dense(1, activation = None)(Y)

	model = Model(inputs = [X, X_gender], outputs = Y)

	print("Created a new model.")

	return model

if(__name__ == "__main__"):
	m = load_model()