import numpy
import pandas as pd
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

# load dataset

#fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
#evaluate model with standardized dataset

# load dataset
dataframe = pd.read_csv("all-data-10.csv", delim_whitespace=False, header=None)
dataset = dataframe.values
days = 9
n1 = 24*days+1
n2 = n1+24
n3 = 80
# split into input (X) and output (Y) variables (training data)
X = dataset[0:n3,1:n1]
Y = dataset[0:n3,n1:n2]

# split into input (A) and output (B) variables (test data)
A = dataset[n3:,1:n1]
B = dataset[n3:,n1:n2]


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
	model.add(Dense(168, init='normal'))
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

# for i in xrange(len(rounded)):
# 	plt.plot([b for b in xrange(24)], rounded[i])
# 	plt.plot([b for b in xrange(24)], rounded[i])
# 	plt.show()

# serialize model to JSON
# model_json = model.to_json()
# with open("model.json", "w") as json_file:
#     json_file.write(model_json)
# serialize weights to HDF5
# model.save_weights("model.h5")
# print("Saved model to disk")
#  
# load json and create model
# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# load weights into new model
# loaded_model.load_weights("model.h5")
# print("Loaded model from disk")
#  
# evaluate loaded model on test data
# loaded_model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])
# score = loaded_model.evaluate(X, Y, verbose=0)
# print "%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100)

#evaluate model with standardized dataset
#estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=100, batch_size=5, verbose=0)
# estimator = KerasRegressor(build_fn=larger_model, nb_epoch=100, batch_size=5, verbose=0)
# 
# kfold = KFold(n_splits=10, random_state=seed)
# results = cross_val_score(estimator, X, Y, cv=kfold)
# print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))


