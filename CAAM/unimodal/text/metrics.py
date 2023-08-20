import numpy as np
import sklearn.metrics

from load_data import load_development_data
from load_model import load_model


import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

if __name__ == "__main__":

	model = load_model()
	model.load_weights('optimal_weights.h5')

	X_dev, Y_dev, X_dev_gender = load_development_data()

	model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])

	dev_Y_hat = model.predict([X_dev, X_dev_gender])

	Y_dev = np.array(Y_dev)
	dev_Y_hat = dev_Y_hat.reshape((Y_dev.shape[0],))

	RMSE = np.sqrt(sklearn.metrics.mean_squared_error(Y_dev, dev_Y_hat))
	MAE = sklearn.metrics.mean_absolute_error(Y_dev, dev_Y_hat)
	EVS = sklearn.metrics.explained_variance_score(Y_dev, dev_Y_hat)

	print('RMSE :', RMSE)
	print('MAE :', MAE)
	print('EVS :', EVS)

	with open('regression_metrics.txt', 'w') as f:
		f.write('RMSE\t:\t' + str(RMSE) + '\nMAE\t\t:\t' + str(MAE) + '\nEVS\t\t:\t' + str(EVS))