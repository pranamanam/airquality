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

X = ds[:,1:73]
Y = ds[:,73:96]


# create model
model = Sequential()
model.add(Dense(72, input_dim=72, init='normal', activation='relu'))
model.add(Dense(23, init='normal'))

# Compile model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X, Y, nb_epoch=150, batch_size=10)

# evaluate the model
scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
