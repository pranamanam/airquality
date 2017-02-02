import numpy
import pandas as pd
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense
from keras.layers import LSTM
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from pylab import * 

# load dataset

#fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
#evaluate model with standardized dataset

# load dataset
dataframe = pd.read_csv("all-data-10.csv", delim_whitespace=False, header=None)
dataset = dataframe.values
days = 9

split = 0.67 #percent of data for training
n1 = 24*days+1 #input hours 
n2 = n1+24 
n3 = int(len(dataset)*split) #var to split data

# input (X) and output (Y) variables (training data -- will be split in fit function)
X = dataset[:,0:n1]
Y = dataset[:,n1:n2]

# split into input (A) and output (B) variables (test data)
A = dataset[n3:,0:n1]
B = dataset[n3:,n1:n2]



#MODELS
#-----------------------------------------------------------------------------------------
# Base model with no hidden layers
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(n1, input_dim=n1, init='normal', activation='relu'))
	model.add(Dense(24, init='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
	return model

#Model with hidden layers
def larger_model():
	# create model
	model = Sequential()
	model.add(Dense(n1, input_dim=n1, init='normal', activation='relu'))
	model.add(Dense(216, init='normal'))
	model.add(Dense(24, init='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
	return model	
	
#-----------------------------------------------------------------------------------------

# Choose which model to run
model = larger_model()

# Fit the model
history = model.fit(X, Y, validation_split=0.33, nb_epoch=500, batch_size=10)

# summarize history for loss and plot MSE
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('MSE Model Evaluation')
plt.ylabel('Mean Squared Error')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# calculate predictions
predictions = model.predict(A)
rounded = []
for arr in predictions:
	rounded.append([round(val) for val in arr])

#Show graphs
for i in xrange(len(rounded)):
	plt.title('Beijing Air Quality Prediction')
	pred, = plt.plot([b+1 for b in xrange(24)], rounded[i], label = 'Predicted Air Quality')
	act, = plt.plot([b+1 for b in xrange(24)], B[i], label = 'Actual Air Quality')
	plt.legend([pred, act], ['Predicted Air Quality', 'Actual Air Quality'])
	plt.xlim(1,24)
	plt.xticks(xrange(24))
	plt.ylabel('Air Quality (ug/m^3)')
	plt.xlabel('Hour')
	plt.show()


