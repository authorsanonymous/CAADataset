import numpy as np
import sklearn.metrics

from load_data import load_test_data
from keras.models import load_model


import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

if __name__ == "__main__":

	model = load_model('min_model.h5')

	X_test, Y_test, X_test_gender = load_test_data()

	model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])

	Y_hat_test = model.predict([X_test, X_test_gender])

	Y_test = np.array(Y_test)
	Y_hat_test = Y_hat_test.reshape((Y_test.shape[0],))

	RMSE = np.sqrt(sklearn.metrics.mean_squared_error(Y_test, Y_hat_test))
	MAE = sklearn.metrics.mean_absolute_error(Y_test, Y_hat_test)
	EVS = sklearn.metrics.explained_variance_score(Y_test, Y_hat_test)

	print('RMSE :', RMSE)
	print('MAE :', MAE)
	print('EVS :', EVS)

	with open('regression_metrics.txt', 'w') as f:
		f.write('RMSE\t:\t' + str(RMSE) + '\nMAE\t\t:\t' + str(MAE) + '\nEVS\t\t:\t' + str(EVS))