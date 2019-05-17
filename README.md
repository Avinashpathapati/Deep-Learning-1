# CNN Project 1
Comparing different settings of CNN for image classification

To run the baseline model:

python cnn_dl.py --m 'cust' --opt "adam" --data_aug 'true' --act 'relu' --data_reg 'true'

without regularization:

python cnn_dl.py --m 'cust' --opt "adam" --data_aug 'true' --act 'relu' --data_reg 'false'

with augmentation:

python cnn_dl.py --m 'cust' --opt "adam" --data_aug 'true' --act 'relu' --data_reg 'true'

without augmentation:

python cnn_dl.py --m 'cust' --opt "adam" --data_aug 'false' --act 'relu' --data_reg 'true'

for adam optimizer ;

python cnn_dl.py --m 'cust' --opt "adam" --data_aug 'true' --act 'relu' --data_reg 'true'

rmsprop optimizer :

python cnn_dl.py --m 'cust' --opt "rms" --data_aug 'true' --act 'relu' --data_reg 'true'

sgd optimizer :

python cnn_dl.py --m 'cust' --opt "sgd" --data_aug 'true' --act 'relu' --data_reg 'true'

relu activation:

python cnn_dl.py --m 'cust' --opt "adam" --data_aug 'true' --act 'relu' --data_reg 'true'

elu activation:

python cnn_dl.py --m 'cust' --opt "adam" --data_aug 'true' --act 'elu' --data_reg 'true'

sigmoid activation:

python cnn_dl.py --m 'cust' --opt "adam" --data_aug 'true' --act 'sig' --data_reg 'true'

resnet:

python cnn_dl.py --m 'resnet' --opt "adam" --data_aug 'true' --act 'relu' --data_reg 'true'

vgg:

python cnn_dl.py --m 'vgg' --opt "adam" --data_aug 'true' --act 'relu' --data_reg 'true'

alexnet:

python cnn_dl.py --m 'alexnet' --opt "adam" --data_aug 'true' --act 'relu' --data_reg 'true'



