from os import path

from keras.models import Model


import numpy as np

from unimodal.feature_extraction.load_data import load_training_data, load_test_data
from unimodal.feature_extraction.load_data import load_model


if(path.exists('optimal_weights.h5')):
	model = load_model()
	model.load_weights('optimal_weights.h5')

else:
	model = load_model()
	model.compile(optimizer='adam', loss='mse', metrics = ['mae'])


new_model = Model(inputs = model.inputs, outputs = model.layers[-5].output)

X_train, y_train, X_train_gender = load_training_data()
X_test, Y_test, X_test_gender = load_test_data()


Y_train_pred = new_model.predict([X_train, X_train_gender])
Y_test_pred = new_model.predict([X_test, X_test_gender])

np.save('./transcript_training.npy', Y_train_pred)
np.save('./transcript_test.npy', Y_test_pred)