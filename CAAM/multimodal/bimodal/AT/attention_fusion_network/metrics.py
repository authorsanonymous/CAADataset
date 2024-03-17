import numpy as np
import sklearn.metrics




import os

from acoustic_X_text.attention_fusion_network.load_data import load_test_data,load_training_data
from acoustic_X_text.attention_fusion_network.load_model import load_model

os.environ["CUDA_VISIBLE_DEVICES"]="1"

if __name__ == "__main__":

	model = load_model()
	model.load_weights('optimal_weights.h5')


	test_FORMANT, test_facial_X_pose, test_gaze_X_action, test_transcript, test_Y, test_X_gender = load_test_data()
	test_transcript = test_transcript.reshape(len(test_transcript), -1)

	model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])

	test_Y_hat = model.predict([test_FORMANT, test_transcript, test_X_gender])

	test_Y = np.array(test_Y)
	test_Y_hat = test_Y_hat.reshape((test_Y.shape[0],))

	RMSE = np.sqrt(sklearn.metrics.mean_squared_error(test_Y, test_Y_hat))
	MAE = sklearn.metrics.mean_absolute_error(test_Y, test_Y_hat)
	EVS = sklearn.metrics.explained_variance_score(test_Y, test_Y_hat)

	print('RMSE :', RMSE)
	print('MAE :', MAE)
	print('EVS :', EVS)

	with open('regression_metrics.txt', 'w') as f:
		f.write('RMSE\t:\t' + str(RMSE) + '\nMAE\t\t:\t' + str(MAE) + '\nEVS\t\t:\t' + str(EVS))