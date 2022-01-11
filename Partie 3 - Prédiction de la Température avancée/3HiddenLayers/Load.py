from math import sqrt
from numpy import concatenate
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import model_from_json
from matplotlib import pyplot
import numpy
import os

#-------------------------------------------------------------------------------------------
# Prepare New Data to make Predictions------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

# load dataset
dataset = read_csv('sys21.csv', header=0, index_col=0)
values = dataset.values
# integer encode direction
encoder = LabelEncoder()
values[:,4] = encoder.fit_transform(values[:,4])
temperatureVal = values[:,0]
pyplot.title('Temperature Values')
pyplot.plot(temperatureVal, label='T')
pyplot.legend()
pyplot.savefig('temperature.png')
pyplot.show()
# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)
# drop columns we don't want to predict
reframed.drop(reframed.columns[[9,10,11,12,13,14,15]], axis=1, inplace=True)


# use all values to make predictions
values = reframed.values
nTaille = int(len(values))
#future = values[nTaille:, :]
# split into input and outputs
values_X, values_y = values[:, :-1], values[:, -1]
#future_X, future_y = future[:, :-1], future[:, -1]
# reshape input to be 3D [samples, timesteps, features]
values_X = values_X.reshape((values_X.shape[0], 1, values_X.shape[1]))
#future_X = future_X.reshape((future_X.shape[0], 1, future_X.shape[1]))
#-------------------------------------------------------------------------------------------
# Import a trained model from JSON and HDF5 files-------------------------------------------
#-------------------------------------------------------------------------------------------
# load json and create model
json_file = open('tempModel.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("tempModel.h5")
print("Loaded model from disk SUCCESS ..... ")
#loaded_model.summary()
#-------------------------------------------------------------------------------------------
# Evaluate loaded model by making predictions and calculating the RMSE,--------------------- 
# then comparing it to the one calculated on the SAVE program-------------------------------
#-------------------------------------------------------------------------------------------
# Make a prediction
yhat = loaded_model.predict(values_X, batch_size=72)
values_X = values_X.reshape((values_X.shape[0], values_X.shape[2]))
# invert scaling for forecast
inv_yhat = concatenate((yhat, values_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
values_y = values_y.reshape((len(values_y), 1))
inv_y = concatenate((values_y, values_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]

# calculate RMSE for test 
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('RMSE = %.3f' % rmse)

# Future predictions based on previous data
# Compare Original and predicted values
pyplot.title('Predictions')
pyplot.plot(temperatureVal, label='Original')
pyplot.plot([None for _ in range(len(temperatureVal))] + [x for x in inv_yhat], label='Future Predictions')
pyplot.legend()
pyplot.savefig('predictAll.png')
pyplot.show()

# Ce sont pas les memes valeurs afficher, Si on les comparent on trouvera la différence
pyplot.title('Comparing predictions & Real Data')
pyplot.plot(temperatureVal, label='Original')
pyplot.plot(inv_yhat, label='Future Predictions')
pyplot.legend()
pyplot.savefig('predictionCompare.png')
pyplot.show()

