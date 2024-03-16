from keras.models import load_model, Model
from load_data.load_FORMANT_features import load_training_data, load_test_data
import numpy as np
from models.FORMANT import *
import os

os.environ['VISIBLE_CUDA_DEVICES'] = '0'

FORMANT_features_model = load_model(location = None)
FORMANT_features_extractor = Model(inputs = FORMANT_features_model.inputs, outputs = FORMANT_features_model.layers[1].output)

X_train, Y_train, X_train_gender = load_training_data()
X_test, Y_test, X_test_gender = load_test_data()

X_train_encoding = FORMANT_features_extractor.predict([X_train, X_train_gender], batch_size = X_train.shape[0])
np.save('./FORMANT_features_training_encoding.npy', X_train_encoding)

X_test_encoding = FORMANT_features_extractor.predict([X_test, X_test_gender], batch_size = X_test.shape[0])
np.save('./FORMANT_features_test_encoding.npy', X_test_encoding)