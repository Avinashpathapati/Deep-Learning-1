import os
import numpy as np
import cv2
import pickle
import scipy
from datetime import *
from optparse import OptionParser
import matplotlib.pyplot as plt

import sys

# Load from file
def ret_acc_load_file(file):
	with open(file, 'rb') as file:  
	    pickle_history = pickle.load(file)
	    mod_acc_list_1 = [i*100 for i in pickle_history['val_acc'] ]
	    mod_tr_acc_list_1 = [i*100 for i in pickle_history['acc'] ]
	    return mod_acc_list_1, mod_tr_acc_list_1

if __name__ == "__main__":
	val_acc_1, tr_acc_1 = ret_acc_load_file('trainHistoryDict_2019-05-01-10-24-01-Adam')
	val_acc_2, tr_acc_2 = ret_acc_load_file('trainHistoryDict_2019-05-08-20-30-25-vgg-prot-2')
	val_acc_3, tr_acc_3 = ret_acc_load_file('trainHistoryDict_2019-05-05-12-45-24-adam-relu-resnet-prot-2')
	print(val_acc_1[124])
	print(val_acc_2[124])
	print(val_acc_3[124])

	#val_acc_3 = ret_acc_load_file('trainHistoryDict_2019-05-03-17-14-09-elu')
	#val_acc_3 = ret_acc_load_file('trainHistoryDict_2019-05-01-18-22-05-sgd')
	print('hello')
	ep = range(1,126)
	plt.plot(ep, val_acc_1)
	plt.plot(ep, val_acc_2)
	plt.plot(ep, val_acc_3)
	#plt.plot(ep,val_acc_3)
	plt.ylabel('Accuracy')
	plt.xlabel('Epochs')
	plt.title('Validation Accuracy Results')
	plt.legend(['Baseline', 'vgg16','resnet20'], loc='upper left')
	plt.show()


    
