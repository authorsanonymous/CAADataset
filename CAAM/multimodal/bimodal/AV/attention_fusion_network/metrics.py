import numpy as np
import sklearn.metrics

from acoustic_X_visual.attention_fusion_network.load_data import load_training_data, load_test_data
from acoustic_X_visual.attention_fusion_network.load_model import load_model

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

if __name__ == "__main__":

	model = load_model()
	model.load_weights('optimal_weights.h5')

	test_FORMANT, test_facial,test_Y, test_X_gender = load_test_data()
gaze
	model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mean_absolute_error'])

	test_Y_hat = model.predict([test_FORMANT, test_facial, test_X_gender])

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