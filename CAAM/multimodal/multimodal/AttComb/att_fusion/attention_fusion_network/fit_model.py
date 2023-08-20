

from attention_fusion_network.load_data import load_training_data, load_test_data


import numpy as np
import os
from os import path

import random

from attention_fusion_network.load_model import load_model

os.environ["CUDA_VISIBLE_DEVICES"] = "0"



if(path.exists('training_progress.csv')):
	progress = np.loadtxt('training_progress.csv', delimiter=',').tolist()

else:
	progress = []

if(path.exists('optimal_weights.h5')):
	model = load_model()
	model.load_weights('optimal_weights.h5')
	model.compile(optimizer='adam', loss='mse', metrics = ['mae'])

else:
	model = load_model()
	model.compile(optimizer='adam', loss='mse', metrics = ['mae'])

training_FORMANT, training_facial, training_transcript, training_Y, training_X_gender = load_training_data()
test_FORMANT, test_facial, test_transcript, test_Y, test_X_gender = load_test_data()

	
min_rmse_test = 50
min_mae_test = 6

current_epoch_number = 1
total_epoch_count = 1


no_of_epochs = 300

m = training_FORMANT.shape[0]
batch_size_list = list(range(1, m))

min_epoch = None


while(current_epoch_number < no_of_epochs):

	print(no_of_epochs - current_epoch_number, "epochs to go.")


	batch_size = 24
	print("Batch size is", batch_size)
	
	hist = model.fit([training_FORMANT, training_facial,  training_X_gender, training_transcript], training_Y, batch_size = batch_size, epochs = 1)
	mse_train = hist.history['loss'][-1]
	mse_test, mae_test = model.evaluate([test_FORMANT, test_facial,  test_X_gender, test_transcript], test_Y, batch_size = test_FORMANT.shape[0])

	print(mse_test, mae_test)

	if (mae_test < min_mae_test):
		min_mae_test = mae_test
		min_mse_test = mse_test
		min_epoch = current_epoch_number

		model.save_weights('optimal_weights.h5')
		print("SAVING THE WEIGHTS!" + "*" * 500 + "\n\n")

		np.savetxt('learner_params.txt', np.array([min_rmse_test, min_mae_test, min_epoch, mse_train]), fmt='%.4f')

	current_epoch_number = current_epoch_number + 1


	progress.append([total_epoch_count, mse_train, mse_test, mae_test])
	np.savetxt('training_progress.csv', np.array(progress), fmt='%.4f', delimiter=',')

	total_epoch_count = total_epoch_count + 1

