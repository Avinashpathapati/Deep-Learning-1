from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers
import os
import numpy as np


class cnn_min_batch_GD():

	def __init__(self, shape_inp):
		weight_decay = 1e-4
		self.network = Sequential()
		print(shape_inp)
		self.network.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu', kernel_regularizer=regularizers.l2(weight_decay),input_shape=shape_inp))
		self.network.add(BatchNormalization())
		self.network.add(Dropout(0.2))
		self.network.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))
		self.network.add(keras.layers.Conv2D(64, kernel_size=3, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))
		self.network.add(BatchNormalization())
		self.network.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))
		self.network.add(Dropout(0.2))

		self.network.add(keras.layers.Conv2D(128, kernel_size=3, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))
		self.network.add(BatchNormalization())
		self.network.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))
		self.network.add(Dropout(0.3))

		self.network.add(keras.layers.Flatten())
		self.network.add(keras.layers.Dense(512, activation='relu'))
		self.network.add(keras.layers.Dense(10, activation='softmax'))

if __name__ == "__main__":
	batch_size = 64
	num_classes = 10
	epochs = 30
	save_dir = os.path.join(os.getcwd(), 'saved_models')
	model_name = 'keras_cifar10_trained_model.h5'

	# The data, split between train and test sets:
	(x_train, y_train), (x_test, y_test) = cifar10.load_data()
	y_train = np.reshape(y_train, y_train.shape[0])
	y_test = np.reshape(y_test, y_test.shape[0])
	print('x_train shape:', x_train.shape)
	print(x_train.shape[0], 'train samples')
	print(x_test.shape[0], 'test samples')
	print(y_train.shape, 'label samples')

	

	# Convert class vectors to binary class matrices.
	print(y_test.shape)
	y_train = keras.utils.to_categorical(y_train, num_classes)
	print(y_train.shape)
	y_test = keras.utils.to_categorical(y_test, num_classes)
	mbg = cnn_min_batch_GD(x_train.shape[1:])
	# initiate RMSprop optimizer
	opt = keras.optimizers.rmsprop(lr=0.001, decay=1e-6)

	# Let's train the model using RMSprop
	mbg.network.compile(loss='categorical_crossentropy',
	          optimizer=opt,
	          metrics=['accuracy'])

	#x_train = x_train.reshape(x_train.shape[0], 32, 32, 3)
	x_train = x_train.astype('float32')

	#X_test = x_test.reshape(x_test.shape[0], 32, 32, 3)
	x_test = x_test.astype('float32')
	x_train /= 255
	x_test /= 255
	datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    )
	datagen.fit(x_train)
	mbg.network.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                    steps_per_epoch=x_train.shape[0] // batch_size,epochs=125,\
                    verbose=1,validation_data=(x_test,y_test))
	if not os.path.isdir(save_dir):
		os.makedirs(save_dir)
	model_path = os.path.join(save_dir, model_name)
	mbg.network.save(model_path)
	print('Saved trained model at %s ' % model_path)

	# Score trained model.
	scores = mbg.network.evaluate(x_test, y_test, verbose=1)
	print('Test loss:', scores[0])
	print('Test accuracy:', scores[1])
