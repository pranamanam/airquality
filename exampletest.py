import warnings
import csv
warnings.filterwarnings("ignore")
import multiprocessing
import os, glob
os.environ['THEANO_FLAGS'] = "floatX=float32,openmp=True"
os.environ['OMP_NUM_THREADS'] = str(multiprocessing.cpu_count())
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from sklearn.model_selection import GridSearchCV, KFold
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import pandas as pd
import numpy

seed = 7

numpy.random.seed(seed)

df = pd.read_csv("formatted.csv")

ds = df.values
ds = ds[2:,:]
X = ds[:,1:73]
Y = ds[:,73]

print X, numpy.shape(X)
raw_input("Press Enter to continue...")

# define base mode
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(72, input_dim=72, init='normal', activation='relu'))
	model.add(Dense(1, init='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

# fix random seed for reproducibility
# seed = 7
# numpy.random.seed(seed)
# evaluate model with standardized dataset
# estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=100, batch_size=5, verbose=0)
# 
# kfold = KFold(n_splits=10, random_state=seed)
# results = cross_val_score(estimator, X, Y, cv=kfold)
# print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

#create model

model = Sequential()
model.add(Dense(72, input_dim=72, init='normal', activation='relu'))
model.add(Dense(1, init='normal'))

#Compile model
#Regression -- let's start with MSE"
model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])

#Fit the model
model.fit(X, Y, nb_epoch=150, batch_size=50)

#evaluate the model
scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# serialize model to JSON
#model_json = model.to_json()
#with open("model.json", "w") as json_file:
#    json_file.write(model_json)
# serialize weights to HDF5
#model.save_weights("model.h5")
#print("Saved model to disk")
