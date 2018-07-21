# import modul
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K

class MyNeuralNets:
	@staticmethod
	def build(width, height, depth, classes):
		# inisialisasi model dengan konfigurasi
		# "channel last"
		model = Sequential()
		inputShape = (height, width, depth)
		chanDim = -1

		# make sure channel first tetap jalan
		if K.image_data_format() == 'channel_first':
			inputShape = (depth, height, width)
			chanDim = 1

		# create CNN Model here
		model.add(Conv2D(32, (5, 5), input_shape=inputShape, activation='relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Conv2D(15, (3, 3), activation='relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.2))
		model.add(Flatten())
		model.add(Dense(100, activation='relu'))
		model.add(Dense(55, activation='relu'))
		model.add(Dense(classes, activation='softmax'))

		return model
