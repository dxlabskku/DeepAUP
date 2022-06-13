from sklearn.model_selection import train_test_split
import numpy as np
import glob
import os
import warnings
warnings.filterwarnings(action='ignore')
import argparse
import torch
from sklearn.metrics import classification_report
import pandas as pd

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--loc', type=str, required=True, help='location of data - [ground / buried ]')
parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
parser.add_argument('--model_name', type=str, default='MFCC_nfft4096_winlength4096_hoplength512_nmfcc30', help='model name')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--dropout', default=0.2, help='dropout')
parser.add_argument('--lr', default=0.001, help='learning rate')
parser.add_argument('--epochs', type=int, default=500, help='epoch')
parser.add_argument('--patience', type=int, default=40, help='patience')
parser.add_argument('--workers', type=int, default=10, help='workers')
parser.add_argument('--data_path', type=str, default='/data/minyoung/pipe/2022/kict/features', help='data path')

opt = parser.parse_args()

loc = opt.loc
gpu_id = opt.gpu_id
model_name = opt.model_name
batch_size = opt.batch_size
dropout = opt.dropout
learning_rate = opt.lr
epochs = opt.epochs
patience = opt.patience
workers = opt.workers
data_path = os.path.join(opt.data_path, '{}/{}'.format(loc, model_name))

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

arr_normal = np.load(os.path.join(data_path, "normal/1.npy"))
shape = arr_normal.shape

print('Load data')
normal_paths = glob.glob("{}/normal/*.npy".format(data_path))
leak_paths = glob.glob("{}/leak/*.npy".format(data_path))
print("before remove 30sec data",len(normal_paths),len(leak_paths))

normal_paths = [path for path in normal_paths if np.load(path).shape==shape]
leak_paths = [path for path in leak_paths if np.load(path).shape==shape]
print("after remove 30sec data",len(normal_paths),len(leak_paths))


paths = np.concatenate([normal_paths,leak_paths])
normal_label = np.zeros(shape=(len(normal_paths,)))
leak_label = np.ones(shape=(len(leak_paths,)))
label = np.concatenate([normal_label,leak_label])

X_train, X_test, y_train, y_test = train_test_split(paths,label,stratify=label,test_size=0.2,random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train,y_train,stratify=y_train,test_size=0.16,random_state=42)

import tensorflow as tf 
import numpy as np 
import os
import tensorflow.keras as keras
from tensorflow.keras.layers import add, Reshape, Dense,Input, TimeDistributed, Dropout, Activation, LSTM, Conv2D, Bidirectional, BatchNormalization 
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l1
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau

from tensorflow.keras import datasets, layers, models
import time


# model structure
def block_CNN(filters, ker, inpC): 
    """
    Returns CNN residual blocks
    """
    layer_1 = BatchNormalization()(inpC) 
    act_1 = Activation('relu')(layer_1) 

    conv_1 = Conv2D(filters, (ker, ker), padding = 'same')(act_1) 

    layer_2 = BatchNormalization()(conv_1) 
    act_2 = Activation('relu')(layer_2) 

    conv_2 = Conv2D(filters, (ker, ker), padding = 'same')(act_2) 
    return(conv_2) 



def block_BiLSTM(inpR, filters, rnn_depth):
    """
    Returns LSTM residual blocks
    """
    x = inpR
    for i in range(rnn_depth):
        x_rnn = Bidirectional(LSTM(filters, return_sequences=True))(x)
        x_rnn = Dropout(0.5)(x_rnn) 
        if i > 0 :
            x = add([x, x_rnn])
        else:
            x = x_rnn      
    return x

def model_cred(shape, filters):

    inp = Input(shape= shape, name='input')

    conv2D_2 = Conv2D(filters[0], (3,3), strides = (2,2), padding = 'same')(inp) 
    res_conv_2 = keras.layers.add([block_CNN(filters[0], 3, conv2D_2), conv2D_2])

    conv2D_3 = Conv2D(filters[1], (3,3), strides = (2,2), padding = 'same')(res_conv_2) 
    res_conv_3 = keras.layers.add([block_CNN(filters[1], 3, conv2D_3),conv2D_3])

    shape = K.int_shape(res_conv_3)   
    reshaped = Reshape((shape[1], shape[2]*shape[3]))(res_conv_3)
    res_BIlstm = block_BiLSTM(reshaped, filters = filters[3], rnn_depth = 2)

    UNIlstm = LSTM(filters[3], return_sequences=False)(res_BIlstm)
    UNIlstm = Dropout(dropout)(UNIlstm)  
    UNIlstm = BatchNormalization()(UNIlstm)

    dense_2 = Dense(filters[3], kernel_regularizer=l1(learning_rate)
                    , activation='relu')(UNIlstm)
    dense_2 = BatchNormalization()(dense_2)
    dense_2 = Dropout(dropout)(dense_2)

    dense_3 = Dense(1, kernel_regularizer=l1(learning_rate), activation='sigmoid')(dense_2)

    out_model = Model(inputs=inp, outputs=dense_3)
    return out_model 

print('\n Create Model')
model = model_cred(shape=(shape[1], shape[0], 1), filters = [8, 16, 32, 64, 128, 256])

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self,paths:list,labels,batch_size:int,shuffle=True):
    # path:str,imagenames,masknames,batch_size:int,imgrow:int,imgcol:int,shuffle=True):
        # imagePath 가 './seg/train/' 일 것 
        self.paths = paths 
        self.labels = labels 
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.datalen = len(paths)
        self.indexes = None 

        image = np.load(self.paths[0]).transpose((1,0))
        self.row = image.shape[0]
        self.col = image.shape[1]

        self.on_epoch_end()
        
    def __getitem__(self,index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        paths = self.paths[indexes]
        labels = self.labels[indexes]
        
        npImages = np.zeros((self.batch_size,self.row,self.col,1),dtype=np.float32)
        for idx,path in enumerate(paths):
            tmp = np.load(path).transpose((1,0))
            tmp = tmp.reshape(self.row,self.col,1)
            npImages[idx] = tmp
            
        return npImages,labels
        
    def on_epoch_end(self):
        self.indexes = np.arange(self.datalen)
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __len__(self):
        return int(np.floor(self.datalen/self.batch_size))

print('\n Data generation...')
train_generator = DataGenerator(X_train,y_train,batch_size,shuffle=True)
valid_generator = DataGenerator(X_valid,y_valid,batch_size,shuffle=True)

model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.binary_crossentropy, metrics=['accuracy',tf.keras.losses.binary_crossentropy])
callback_list = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=patience,restore_best_weights=True)
    ]
print('\n Model training...')
start = time.time()
with tf.device('/device:GPU:{}'.format(gpu_id)):
    history = model.fit_generator(train_generator,epochs=epochs,
                                    use_multiprocessing=True,workers=workers
                                ,validation_data=valid_generator
                                    ,callbacks=callback_list)
print('Train time: {}'.format(time.time() - start))

# save model
import pickle
print('\n Model save...')
model.save("models_history/%s.h5"%(model_name))
with open("models_history/%s.bin"%(model_name),"wb") as f:
    pickle.dump(history.history,f,pickle.HIGHEST_PROTOCOL)


# test
print('\n Model test...')
test_generator = DataGenerator(X_test,y_test,1,shuffle=False)
start_t  = time.time()
output = model.predict_generator(test_generator)
print('Test time: {}'.format(time.time() - start_t))
output = (output>0.5).astype(np.int32)

print(model.evaluate_generator(test_generator))



import pandas as pd
report = classification_report(y_test,output.reshape((-1,)),digits=6, output_dict = True)
result = pd.DataFrame(report).transpose()
print('\n save the result')
result.to_csv('result_{}_{}.csv'.format(loc, model_name), index = False)
