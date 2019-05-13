from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras
from keras.datasets import cifar10
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, AveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D,Input,GlobalAveragePooling2D
from keras.applications.resnet50 import ResNet50
from keras import regularizers
import tensorflow as tf
from keras import applications
import os
import numpy as np
import cv2
import pickle
import scipy
from datetime import *
from optparse import OptionParser


from keras import backend as K
import sys
from six.moves import cPickle

def load_data(cur):
	"""Loads CIFAR10 dataset.
	# Returns
		Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
	"""
	
	#path = os.path.join(os.getcwd(),'cifar-10-batches-py')
	path = '/content/gdrive/My Drive/cifar-10-batches-py'

	num_train_samples = 10000

	x_train = np.empty((num_train_samples, 197, 197, 3), dtype='uint8')
	y_train = np.empty((num_train_samples,), dtype='uint8')
	print('processing',cur)
	fpath = os.path.join(path, 'data_batch_' + str(cur))
	(x_train[0: 10000, :, :, :],
	y_train[0: 10000]) = load_batch(fpath)

	fpath = os.path.join(path, 'test_batch')
	x_test, y_test = load_batch(fpath)

	y_train = np.reshape(y_train, (len(y_train), 1))
	y_test = np.reshape(y_test, (len(y_test), 1))

	# if K.image_data_format() == 'channels_last':
	# 	print('true#')
	#     x_train = x_train.transpose(0, 2, 3, 1)
	#     x_test = x_test.transpose(0, 2, 3, 1)

	return (x_train, y_train), (x_test, y_test)

def load_batch(fpath, label_key='labels'):
	"""Internal utility for parsing CIFAR data.
	# Arguments
		fpath: path the file to parse.
		label_key: key for label data in the retrieve
			dictionary.
	# Returns
		A tuple `(data, labels)`.
	"""
	with open(fpath, 'rb') as f:
		if sys.version_info < (3,):
			d = cPickle.load(f)
		else:
			d = cPickle.load(f, encoding='bytes')
			# decode utf8
			d_decoded = {}
			for k, v in d.items():
				d_decoded[k.decode('utf8')] = v
			d = d_decoded
	data = d['data']
	labels = d[label_key]

	data = data.reshape(data.shape[0], 32, 32, 3)
	data = resize_img(data)
	return data, labels


def resize_img(data):

	data_upscale = np.zeros((data.shape[0], 197, 197, 3))
	for i, img in enumerate(data):
		large_img = cv2.resize(img, dsize=(197, 197), interpolation=cv2.INTER_CUBIC)
		data_upscale[i] = large_img

	return data_upscale

class cnn_min_batch_GD():


	def resnet_layer(self, inputs,num_filters=16,kernel_size=3,strides=1,activation='relu',batch_normalization=True,conv_first=True):

		conv = keras.layers.Conv2D(num_filters,kernel_size=kernel_size,strides=strides,padding='same',
			kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(1e-4))
		x = inputs
		if conv_first:
			x = conv(x)
			if batch_normalization:
				x = BatchNormalization()(x)
			if activation is not None:
				x = Activation(activation)(x)
		else:
			if batch_normalization:
				x = BatchNormalization()(x)
			if activation is not None:
				x = Activation(activation)(x)
			x = conv(x)
		return x

	def vgg(self, input_shape):
		   # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.
			
		print('vgg')
		print(input_shape)
		model = Sequential()
		weight_decay = 0.0005

		model.add(Conv2D(64, (3, 3), padding='same',
						 input_shape=input_shape,kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(Activation('relu'))
		model.add(BatchNormalization())
		model.add(Dropout(0.3))

		model.add(Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(Activation('relu'))
		model.add(BatchNormalization())

		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(Activation('relu'))
		model.add(BatchNormalization())
		model.add(Dropout(0.4))

		model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(Activation('relu'))
		model.add(BatchNormalization())

		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(Activation('relu'))
		model.add(BatchNormalization())
		model.add(Dropout(0.4))

		model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(Activation('relu'))
		model.add(BatchNormalization())
		model.add(Dropout(0.4))

		model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(Activation('relu'))
		model.add(BatchNormalization())

		model.add(MaxPooling2D(pool_size=(2, 2)))


		model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(Activation('relu'))
		model.add(BatchNormalization())
		model.add(Dropout(0.4))

		model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(Activation('relu'))
		model.add(BatchNormalization())
		model.add(Dropout(0.4))

		model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(Activation('relu'))
		model.add(BatchNormalization())

		model.add(MaxPooling2D(pool_size=(2, 2)))


		model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(Activation('relu'))
		model.add(BatchNormalization())
		model.add(Dropout(0.4))

		model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(Activation('relu'))
		model.add(BatchNormalization())
		model.add(Dropout(0.4))

		model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(Activation('relu'))
		model.add(BatchNormalization())

		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.5))

		model.add(Flatten())
		model.add(Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(Activation('relu'))
		model.add(BatchNormalization())

		model.add(Dropout(0.5))
		model.add(Dense(10))
		model.add(Activation('softmax'))

		return model

	def inception(self):
		input_img = Input(shape = (32, 32, 3))
		l_1 = Conv2D(64, (1,1), padding='same', activation='relu')(input_img)
		l_1 = Conv2D(64, (3,3), padding='same', activation='relu')(l_1)
		l_2 = Conv2D(64, (1,1), padding='same', activation='relu')(input_img)
		l_2 = Conv2D(64, (5,5), padding='same', activation='relu')(l_2)
		l_3 = MaxPooling2D((3,3), strides=(1,1), padding='same')(input_img)
		l_3 = Conv2D(64, (1,1), padding='same', activation='relu')(l_3)
		output = keras.layers.concatenate([l_1, l_2, l_3], axis = 3)
		output = Flatten()(output)
		out    = Dense(10, activation='softmax')(output)
		model = Model(inputs = input_img, outputs = out)
		return model

	def Alexnet(self, input_shape):
		#K.set_image_dim_ordering('th')
		model = Sequential()
		# model.add(Conv2D(96,(3,3),strides=(1,1),input_shape=(32,32,3),padding='valid',activation='relu',kernel_initializer='uniform'))
		model.add(Conv2D(96,(11,11),strides=(1,1),input_shape=input_shape,padding='same',activation='relu',kernel_initializer='uniform'))
		model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
		model.add(Conv2D(256,(5,5),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
		model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
		model.add(Conv2D(384,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
		model.add(Conv2D(384,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
		model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
		model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
		model.add(Flatten())
		model.add(Dense(4096,activation='relu'))
		model.add(Dropout(0.5))
		model.add(Dense(4096,activation='relu'))
		model.add(Dropout(0.5))
		model.add(Dense(10,activation='softmax'))
		return model

	def resnet_v1(self, input_shape, depth, num_classes=10):

		"""ResNet Version 1 Model resnet_layer [a]

		Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
		Last ReLU is after the shortcut connection.
		At the beginning of each stage, the feature map size is halved (downsampled)
		by a convolutional layer with strides=2, while the number of filters is
		doubled. Within each stage, the layers have the same number filters and the
		same number of filters.
		Features maps sizes:
		stage 0: 32x32, 16
		stage 1: 16x16, 32
		stage 2:  8x8,  64
		The Number of parameters is approx the same as Table 6 of [a]:
		ResNet20 0.27M
		ResNet32 0.46M
		ResNet44 0.66M
		ResNet56 0.85M
		ResNet110 1.7M

		# Arguments
			input_shape (tensor): shape of input image tensor
			depth (int): number of core convolutional layers
			num_classes (int): number of classes (CIFAR10 has 10)

		# Returns
			model (Model): Keras model instance
		"""
		if (depth - 2) % 6 != 0:
			raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
		# Start model definition.
		num_filters = 16
		num_res_blocks = int((depth - 2) / 6)

		inputs = Input(shape=input_shape)
		x = self.resnet_layer(inputs=inputs)
		# Instantiate the stack of residual units
		for stack in range(3):
			for res_block in range(num_res_blocks):
				strides = 1
				if stack > 0 and res_block == 0:  # first layer but not first stack
					strides = 2  # downsample
				y = self.resnet_layer(inputs=x,
								 num_filters=num_filters,
								 strides=strides)
				y = self.resnet_layer(inputs=y,
								 num_filters=num_filters,
								 activation=None)
				if stack > 0 and res_block == 0:  # first layer but not first stack
					# linear projection residual shortcut connection to match
					# changed dims
					x = self.resnet_layer(inputs=x,
									 num_filters=num_filters,
									 kernel_size=1,
									 strides=strides,
									 activation=None,
									 batch_normalization=False)
				x = keras.layers.add([x, y])
				x = Activation('relu')(x)
			num_filters *= 2

		# Add classifier on top.
		# v1 does not use BN after last shortcut connection-ReLU
		x = AveragePooling2D(pool_size=8)(x)
		y = Flatten()(x)
		outputs = Dense(num_classes,
						activation='softmax',
						kernel_initializer='he_normal')(y)

		# Instantiate model.
		model = Model(inputs=inputs, outputs=outputs)
		return model
	
	def __init__(self, shape_inp, modelType, act_fun, data_reg):
		if modelType == 'resnet':
			print('resnet')
			self.network = self.resnet_v1(input_shape=shape_inp, depth=20)
			# base_model = ResNet50(include_top=False, weights=None, input_shape=(140,140,3), pooling=None)
			# x = keras.layers.GlobalAveragePooling2D()(base_model.output)
			# output = keras.layers.Dense(10, activation='softmax')(x)
			# self.network = keras.models.Model(inputs=[base_model.input], outputs=[output])
		elif modelType == 'incep':
			self.network = self.inception()
			#self.network = applications.inception_v3.InceptionV3(include_top=False, weights=None, input_tensor=None, input_shape=(32,32,3), pooling=None, classes=10)
		elif modelType == 'vgg':
			#self.network = applications.vgg16.VGG16( weights=None, include_top=True, classes=10, input_shape=(32,32,3))
			self.network = self.vgg(input_shape = shape_inp)
		elif modelType == 'alexnet':
			self.network = self.Alexnet(input_shape = shape_inp)
		else :
			if act_fun == 'relu':
				activation = 'relu'
			elif act_fun == 'elu':
				activation = 'elu'
			else:
				activation = 'sigmoid'

			weight_decay = 1e-4
			self.network = Sequential()
			if data_reg == 'true':
				self.network.add(keras.layers.Conv2D(32, kernel_size=3, activation=activation, kernel_regularizer=regularizers.l2(weight_decay),input_shape=shape_inp))
				self.network.add(BatchNormalization())
				self.network.add(Dropout(0.2))
				self.network.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))
				self.network.add(keras.layers.Conv2D(64, kernel_size=3, activation=activation, kernel_regularizer=regularizers.l2(weight_decay)))
				self.network.add(BatchNormalization())
				self.network.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))
				self.network.add(Dropout(0.2))

				self.network.add(keras.layers.Conv2D(128, kernel_size=3, activation=activation, kernel_regularizer=regularizers.l2(weight_decay)))
				self.network.add(BatchNormalization())
				self.network.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))
				self.network.add(Dropout(0.3))

				self.network.add(keras.layers.Flatten())
				self.network.add(keras.layers.Dense(512, activation=activation))
				self.network.add(keras.layers.Dense(10, activation='softmax'))

			else:
				self.network.add(keras.layers.Conv2D(32, kernel_size=3, activation=activation,input_shape=shape_inp))
				self.network.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))
				self.network.add(keras.layers.Conv2D(64, kernel_size=3, activation=activation))
				self.network.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))

				self.network.add(keras.layers.Conv2D(128, kernel_size=3, activation=activation))
				self.network.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))

				self.network.add(keras.layers.Flatten())
				self.network.add(keras.layers.Dense(512, activation=activation))
				self.network.add(keras.layers.Dense(10, activation='softmax'))

if __name__ == "__main__":
	#commandline arguments reading
	parser = OptionParser()
	parser.add_option("-m", "--model",
				  action="store", type="string", dest="model")

	parser.add_option("-o", "--opt",
				  action="store", type="string", dest="opt")

	parser.add_option("-a", "--act",
				  action="store", type="string", dest="act")

	parser.add_option("-u", "--data_aug",
				  action="store", type="string", dest="data_aug")

	parser.add_option("-r", "--data_reg",
				  action="store", type="string", dest="data_reg")


	(options, args) = parser.parse_args()


	batch_size = 64
	num_classes = 10
	epochs = 30
	save_dir = os.path.join(os.getcwd(), 'saved_models')
	model_name = 'keras_cifar10_trained_model.h5'

	# The data, split between train and test sets:
	(x_train, y_train), (x_test, y_test) = cifar10.load_data()
	y_train = np.reshape(y_train, y_train.shape[0])
	y_test = np.reshape(y_test, y_test.shape[0])
	print('x_train shape:', x_train.shape[1:])
	print(x_train.shape[0], 'train samples')
	print(x_test.shape[0], 'test samples')

	

	# Convert class vectors to binary class matrices.
	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_test = keras.utils.to_categorical(y_test, num_classes)

	mbg = cnn_min_batch_GD(x_train.shape[1:], options.model, options.act, options.data_reg)
	
	if options.opt == 'sgd':
		opt = keras.optimizers.SGD(lr=0.001, decay=1e-6)
	elif options.opt == 'adam':
		opt = keras.optimizers.Adam(lr=0.001, decay=1e-6)
	else:
		opt = keras.optimizers.RMSprop(lr=0.001, decay=1e-6)

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
	if options.data_aug == 'false':
		history = mbg.network.fit(x_train, y_train,
			  batch_size=batch_size,
			  epochs=125,
			  validation_data=(x_test, y_test),
			  shuffle=True)
	else:
		datagen = ImageDataGenerator(
		rotation_range=15,
		width_shift_range=0.1,
		height_shift_range=0.1,
		horizontal_flip=True,
		)
		datagen.fit(x_train)
		history = mbg.network.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
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

	#saving into pickle file
	with open('/trainHistoryDict_'+datetime.now().strftime("%Y-%m-%d-%H-%M-%S"), 'wb') as file_pi:
		pickle.dump(history.history, file_pi)

