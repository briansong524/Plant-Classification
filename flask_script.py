import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import sys
import random

import numpy as np
import numpy.core.defchararray as np_string
import pandas as pd

import glob

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense, Input
from keras.optimizers import Adam, SGD
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import to_categorical
import keras.backend as K
import tensorflow as tf

from flask import Flask, render_template, request

from skimage.transform import resize as imresize
import imageio


sys.path.append(os.getcwd())#os.path.abspath('/home/bsong/'))


app = Flask(__name__)

path_to_dir = '/home/bsong'
work_dir = path_to_dir+ '/kaggle_plants/'
train_dir = work_dir + 'train/'
test_dir = work_dir + 'Flask'
static_dir = '/static/'
model_path = work_dir + 'model/'

# Dense layers set
def dense_set(inp_layer, n, activation, drop_rate=0.):
	dp = Dropout(drop_rate)(inp_layer)
	dns = Dense(n)(dp)
	bn = BatchNormalization(axis=-1)(dns)
	act = Activation(activation=activation)(bn)
	return act

# Conv. layers set
def conv_layer(feature_batch, feature_map, kernel_size=(3, 3),strides=(1,1), zp_flag=False):
	if zp_flag:
		zp = ZeroPadding2D((1,1))(feature_batch)
	else:
		zp = feature_batch
	conv = Conv2D(filters=feature_map, kernel_size=kernel_size, strides=strides)(zp)
	bn = BatchNormalization(axis=3)(conv)
	act = LeakyReLU(1/10)(bn)
	
	return act

def get_model():
    inp_img = Input(shape=(51, 51, 3))

    # 51
    conv1 = conv_layer(inp_img, 64, zp_flag=False)
    conv2 = conv_layer(conv1, 64, zp_flag=False)
    mp1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(conv2)
    # 23
    conv3 = conv_layer(mp1, 128, zp_flag=False)
    conv4 = conv_layer(conv3, 128, zp_flag=False)
    mp2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(conv4)
    # 9
    conv7 = conv_layer(mp2, 256, zp_flag=False)
    conv8 = conv_layer(conv7, 256, zp_flag=False)
    conv9 = conv_layer(conv8, 256, zp_flag=False)
    mp3 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(conv9)
    # 1
    # dense layers
    flt = Flatten()(mp3)
    ds1 = dense_set(flt, 128, activation='tanh')
    out = dense_set(ds1, 12, activation='softmax')

    model = Model(inputs=inp_img, outputs=out)
    
    # The first 50 epochs are used by Adam opt.
    # Then 30 epochs are used by SGD opt.
    
    #mypotim = Adam(lr=2 * 1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    mypotim = SGD(lr=1 * 1e-1, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                   optimizer=mypotim,
                   metrics=['accuracy'])
    model.summary()

    global graph
    graph = tf.get_default_graph()

    return model

def load_Data():
	print('Importing Data')
	K.clear_session()
	model_ = get_model()
	model_.load_weights(filepath = model_path + 'model_weight_SGD.hdf5')
	
	list_paths = glob.glob(train_dir + '*')
	list_names = [i.replace(train_dir,'') for i in list_paths]
	name2ind = dict(zip(list_names, range(len(list_names))))
	ind2name = dict(zip(range(len(list_names)),list_names))

	list_test = glob.glob(test_dir + static_dir + '*')
	test_dict = dict(zip(range(len(list_test)), list_test))
	print('Done')
	
	return model_, name2ind, ind2name, test_dict

global model_, name2ind, ind2name, test_dict
model_, name2ind, ind2name, test_dict = load_Data()





@app.route('/generate', methods = ['GET'])


def generate_page():

	# model page
	image_num = request.args.get('imageid')
	print(image_num)
	string_output = rng_output(image_num)
	return render_template('generate_page.html', variable=string_output)


def rng_output(image_num):

	if ((image_num == '0') | (image_num == '')):
		rand_ind = random.sample(range(len(test_dict.keys())), 1)[0] #| predefined value from form  
		image_num = list(test_dict.keys())[rand_ind]
	else:
		image_num = int(image_num)

	pred_img = np.array([img_get(test_dict[image_num])])
	#pred_img = np.expand_dims(pred_img, axis = 0)
	#print(pred_img.shape)
	with graph.as_default():
		prob = model_.predict(pred_img)
	pred = prob.argmax(axis=-1)
	string_output = dict()
	string_output['image_num'] = str(image_num)
	string_output['prediction'] = str(ind2name[pred[0]])
	string_output['image_path'] = str(test_dict[image_num]).replace(test_dir, '')
	return string_output
	
def img_reshape(img):
	img = imresize(img, (51, 51, 3))
	return img

def img_get(path):
	img = imageio.imread(path)
	img = img_reshape(img)
	return img
	

if __name__ == '__main__':
	app.run(debug=True, port=8986, host='0.0.0.0')