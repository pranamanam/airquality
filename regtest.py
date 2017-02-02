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
dataframe = pd.read_csv("sliding-days-data-5.csv", delim_whitespace=False, header=None)
dataset = dataframe.values
days = 4

split = 0.9 #percent of data for training
n1 = 24*days+1 #input hours 
n2 = n1+24 
n3 = int(len(dataset)*split) #var to split data

# split into input (X) and output (Y) variables (training data)
X = dataset[0:n3,1:n1]
Y = dataset[0:n3,n1:n2]

# split into input (A) and output (B) variables (test data)
A = dataset[n3:,1:n1]
B = dataset[n3:,n1:n2]

print(len(X))
print(len(A))


#MODELS
#-----------------------------------------------------------------------------------------
# Base model with no hidden layers
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(n1-1, input_dim=n1-1, init='normal', activation='relu'))
	model.add(Dense(24, init='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
	return model

#Model with hidden layers
def larger_model():
	# create model
	model = Sequential()
	model.add(Dense(n1-1, input_dim=n1-1, init='normal', activation='relu'))
	model.add(Dense(n1-2, init='normal'))
	model.add(Dense(24, init='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
	return model	
	
#-----------------------------------------------------------------------------------------

# Choose which model to run
model = larger_model()

# Fit the model
model.fit(X, Y, nb_epoch=500, batch_size=10)

# evaluate the model
scores = model.evaluate(X, Y)
print()
print("Training: %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

scores = model.evaluate(A, B)
print("Test: %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# calculate predictions
predictions = model.predict(A)
rounded = []
for arr in predictions:
	rounded.append([round(val) for val in arr])

#print(rounded)
#print(B)

for i in xrange(len(rounded)/4):
	plt.title('Beijing Air Quality Prediction')
	pred, = plt.plot([b+1 for b in xrange(24)], rounded[i], label = 'Predicted Air Quality')
	act, = plt.plot([b+1 for b in xrange(24)], B[i], label = 'Actual Air Quality')
	plt.legend([pred, act], ['Predicted Air Quality', 'Actual Air Quality'])
	plt.xlim(1,24)
	plt.xticks(xrange(24))
	plt.ylabel('Air Quality (ug/m^3)')
	plt.xlabel('Hour')
	plt.show()


