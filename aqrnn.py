import numpy
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# fix random seed for reproducibility
numpy.random.seed(7)

# load the dataset
dataframe = pd.read_csv("all-data-10.csv", delim_whitespace=False, header=None)
dataset = dataframe.values
dataset = dataset.astype('float32')

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# reshape into X=t and Y=t+1
days = 9

split = 0.67 #percent of data for training
n1 = 24*days+1 #input hours 
n2 = n1+24 
n3 = int(len(dataset)*split) #var to split data

# input (X) and output (Y) variables (training data -- will be split in fit function)
trainX = dataset[0:n3,0:n1]
trainY = dataset[0:n3,n1:n2]

# split into input (A) and output (B) variables (test data)
testX = dataset[n3:,0:n1]
testY = dataset[n3:,n1:n2]

# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_dim=n1))
model.add(Dense(24))
model.compile(loss='mean_squared_error', optimizer='adam')
history = model.fit(trainX, trainY, nb_epoch=100, batch_size=1, verbose=2)

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# invert predictions
# trainPredict = scaler.inverse_transform(trainPredict)
# trainY = scaler.inverse_transform([trainY])
# testPredict = scaler.inverse_transform(testPredict)
# testY = scaler.inverse_transform([testY])

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY, testPredict))
print('Test Score: %.2f RMSE' % (testScore))

print trainY
print trainPredict

print testY
print testPredict

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('MSE Model Evaluation')
plt.ylabel('Mean Squared Error')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

for i in xrange(10):
	plt.title('Beijing Air Quality Prediction')
	pred, = plt.plot([b for b in xrange(24)], testPredict[i], label = 'Predicted Air Quality')
	act, = plt.plot([b for b in xrange(24)], testY[i], label = 'Actual Air Quality')
	plt.legend([pred, act], ['Predicted Air Quality', 'Actual Air Quality'])
	plt.xlim(0,23)
	plt.xticks(xrange(24))
	plt.ylabel('Scaled Air Quality')
	plt.xlabel('Hour')
	plt.show()