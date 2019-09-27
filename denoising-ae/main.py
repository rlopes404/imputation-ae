#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 14:43:17 2019

@author: ramon
"""

import numpy as np
np.random.seed(0)
import pandas as pd

import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split

from keras import callbacks
from keras import regularizers
from keras.layers import Input, Dense
from keras.models import Model


n_layers = 2
encoding_dim = 32
l1_reg = 1e-8
epochs = 50
batch_size = 256
name = './data2.csv'

probs =  np.arange(0.1, 1.0, 0.1)

#reading dataset
df = pd.read_csv(name, sep=';', header=0)
 
#NAN analysis
 
values = df.isna().sum()[1:]/df.shape[0]
#corruption_prob = np.max(values)
 
 
df.dropna(axis=0, inplace=True)
 
df = df.astype('float32')
 
input_dim = df.shape[1]
output_dim = input_dim - 1
 
#splitting dataset
x_train, x_test = train_test_split(df, test_size=0.2, random_state=0)
 
 
#normalizing datasets
mean_values = x_train.mean()
std_values = x_train.std()
 
x_train = (x_train - mean_values)/std_values
x_test = (x_test - mean_values)/std_values
x_train, x_val = train_test_split(x_train, test_size=0.2, random_state=0)
 
#converting from DataFrame to numpy array as Keras accepts numpy arrays
x_train = x_train.values
x_val = x_val.values
x_test = x_test.values
 
probs =  np.arange(0.1, 1.0, 0.1)

def corrupt_data_interval(df, probs):
    n_rows = df.shape[0]
    n_columns = df.shape[1]    
         
    data_augmentation_factor = len(probs)
     
    df = np.repeat(df, data_augmentation_factor, axis=0)
    month_mask = np.ones((df.shape[0], 1))
     
    mask = None
    for p in probs:        
        _mask = np.random.binomial(1, 1-p, size=(n_rows, n_columns-1))      
        if mask is None:
             mask = _mask
        else:            
            mask = np.concatenate((mask, _mask), axis=0)
     
    return df, df*np.concatenate((month_mask, mask), axis=1), mask  

     
def corrupt_data(df, noise_prob=0.5):
    n_rows = df.shape[0]
    n_columns = df.shape[1]    
    month_mask = np.ones((n_rows, 1))    
     
    mask = np.random.binomial(1, 1-noise_prob, size=(n_rows, n_columns-1))    
    return df*np.concatenate((month_mask, mask), axis=1), mask


_x_train, _x_train_corrupted, _x_train_mask = corrupt_data_interval(x_train, probs)
_x_val, _x_val_corrupted, _x_val_mask = corrupt_data_interval(x_val, probs)

#creating autoencoder
layers = [(2**i)*encoding_dim for i in range(1,n_layers+1)]

#encoder layers
input_ts = Input(shape=(input_dim,), dtype='float32')
if len(layers) > 0:
    x = None
    for i, dim in enumerate(reversed(layers)):
        encoder = Dense(dim, activation='relu', activity_regularizer=regularizers.l1(l1_reg))
        x = encoder(x) if i!=0 else encoder(input_ts)
    
    encoded = Dense(encoding_dim, activation='relu', activity_regularizer=regularizers.l1(l1_reg))(x)
    
    for i, dim in enumerate(layers):
        decoder = Dense(dim, activation='relu', activity_regularizer=regularizers.l1(l1_reg))
        x = decoder(x) if i!=0 else decoder(encoded)
        
    decoded = Dense(output_dim)(x) 
else:
    encoded = Dense(encoding_dim, activation='relu', activity_regularizer=regularizers.l1(l1_reg))(input_ts)
    decoded = Dense(output_dim)(encoded)    

autoencoder = Model(input_ts, decoded)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

#training autoencoder

es = callbacks.EarlyStopping(monitor='val_loss', mode='min',  min_delta=0.01, patience=10, restore_best_weights=True, verbose=0)
autoencoder_train = autoencoder.fit(_x_train_corrupted, _x_train[:,1:], validation_data=(_x_val_corrupted, _x_val[:,1:]), epochs=epochs, batch_size=batch_size, shuffle=True, verbose=1, callbacks=[es])

loss = autoencoder_train.history['loss']
val_loss = autoencoder_train.history['val_loss']
x = range(len(loss))
plt.figure()
plt.plot(x, loss, 'bo', label='Training loss')
plt.plot(x, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

#testing the model

def get_rmse(predicted, y, std, mask=None):
    rmse = (predicted - y[:,1:])*std[1:]
    if not (mask is None):
        rmse = rmse*(1-mask) #considering only corrputed columns
    rmse = np.sqrt(np.square(rmse).mean(axis=0))
    return rmse


x_val_predicted = autoencoder.predict(_x_val_corrupted)
rmse_val = get_rmse(x_val_predicted, _x_val, std_values.values, mask=_x_val_mask)
rmse_val_full = get_rmse(x_val_predicted, _x_val, std_values.values, mask=None)
print('RMSE-val: %.2f' %(np.mean(rmse_val)))
print('RMSE-val Full: %.2f' %(np.mean(rmse_val_full)))


for p in probs:
    _x_val, _x_val_corrupted, _x_val_mask = corrupt_data_interval(x_val, [p])
    _x_test, _x_test_corrupted, _x_test_mask = corrupt_data_interval(x_test, [p])
    
    _x_val_predicted = autoencoder.predict(_x_val_corrupted)
    _x_test_predicted = autoencoder.predict(_x_test_corrupted)
    
    _rmse_val = get_rmse(_x_val_predicted, _x_val, std_values.values, mask=_x_val_mask)    
    _rmse_test = get_rmse(_x_test_predicted, _x_test, std_values.values, mask=_x_test_mask)    
    print('RMSE (val, test) %.2f: %.4f \t %.4f' %(p, np.mean(_rmse_val),  np.mean(_rmse_test)))

    
#rmse_test_full = get_rmse(x_test_predicted, x_test, std_values.values, mask=None)

#print('RMSE-test Full: %.2f' %(np.mean(rmse_test_full)))


if True:
#ploting rmse for each variable    
    objects = np.arange(len(rmse_val))
    y_pos = np.arange(len(objects))
    plt.bar(y_pos, rmse_val, align='center', alpha=0.5)
    plt.plot(np.arange(len(objects)), np.repeat(1, len(objects)), 'r')
    plt.xticks(y_pos, objects)
    plt.ylabel('RMSE')
    plt.show()
    
    #what are the variables whose rmse is greater than or equal to 1.0?
    idxs = np.where(rmse_val >= 1.0)
    print(df.columns[idxs])




#autoencoder.predict(x_test)




#TODO: Seq-to-seq auto-encoders, variational autoencoders
#https://blog.keras.io/building-autoencoders-in-keras.html

#df = pd.DataFrame(np.array([[1,2,3],[4,5,6]]), columns=['a','b','c'])


