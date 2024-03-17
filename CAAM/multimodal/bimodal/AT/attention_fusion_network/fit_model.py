from keras.models import Model

from acoustic_X_text.attention_fusion_network.load_data import load_training_data, load_test_data


import keras

import numpy as np
import os
from os import path

import random

from acoustic_X_text.attention_fusion_network.load_model import load_model

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


if(path.exists('learner_params.txt')):
	learner_params = np.loadtxt('learner_params.txt')
	min_loss_test = learner_params[0]
	min_mae = learner_params[1]
	prev_loss_test = learner_params[4]
	loss_test = [learner_params[2], learner_params[3]]

	current_epoch_number = int(learner_params[6])
	total_epoch_count = int(learner_params[7]) + 1

else:
	min_loss_test = 70
	min_mae = 7
	prev_loss_test = 70
	loss_test = [70, 70]

	current_epoch_number = 1
	total_epoch_count = 1

increase_count = 0
increase_count_threshold = 10
no_of_downward_epochs = 50000

m = training_FORMANT.shape[0]
batch_size_list = list(range(1, m))


# min_epoch = None
while(current_epoch_number < no_of_downward_epochs):

	print(no_of_downward_epochs - current_epoch_number, "epochs to go.")
	prev_loss_test = loss_test[0]
	# batch_size = random.choice(batch_size_list)
	# batch_size = int(m/4.5)
	batch_size = 8
	print("Batch size is", batch_size)
	hist = model.fit([training_FORMANT, training_transcript, training_X_gender], training_Y, batch_size = batch_size, epochs = 1)
	loss_train = hist.history['loss'][-1]
	loss_test = model.evaluate([test_FORMANT, test_transcript, test_X_gender], test_Y, batch_size = test_FORMANT.shape[0])

	print(loss_train, loss_test[0], loss_test[1])

	if(loss_test[0] < min_loss_test):
		min_loss_test = loss_test[0]
		min_mae = loss_test[1]
		model.save_weights('optimal_weights.h5')
		print("SAVING THE WEIGHTS!"+"*"*20+"\n\n")
		increase_count = 0
		current_epoch_number = current_epoch_number + 1

	else:

		if(loss_test[0] >= prev_loss_test):
			increase_count = increase_count + 1
			print("increase_count: ", increase_count)
			print("increase_count_threshold: ", increase_count_threshold)

			if(increase_count >= increase_count_threshold):
				model.load_weights('optimal_weights.h5')
				increase_count = 16
				print("\nGOING BACK TO THE min_model!")
				current_epoch_number = current_epoch_number + 1

		else:
			increase_count = increase_count - 1

			if(increase_count < 0):
				increase_count = 0
				current_epoch_number = current_epoch_number + 1

	progress.append([total_epoch_count, loss_train, loss_test[0], loss_test[1]])
	np.savetxt('training_progress.csv', np.array(progress), fmt='%.4f', delimiter=',')
	np.savetxt('learner_params.txt', np.array([min_loss_test, min_mae, loss_test[0], loss_test[1], prev_loss_test, increase_count, current_epoch_number,total_epoch_count]), fmt='%.4f')

	total_epoch_count = total_epoch_count + 1

