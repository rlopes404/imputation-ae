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
augment_data = True
temporal_split = True

probs =  np.arange(0.1, 1.0, 0.1)

df = pd.read_csv(name, sep=';', header=0)


#values = df.isna().sum()[1:]/df.shape[0]
#corruption_prob = np.max(values)


df.dropna(axis=0, inplace=True)
df = df.astype('float32')

input_dim = df.shape[1]
output_dim = input_dim - 1

#splitting dataset
if temporal_split:
    threshold = int(df.shape[0]*0.8)
    x_train = df.iloc[:threshold, :].copy()
    x_test = df.iloc[threshold:, :].copy()
    
    mean_values = x_train.mean()
    std_values = x_train.std()
    
    
    threshold = int(x_train.shape[0]*0.8)
    x_val = x_train.iloc[threshold:, :].copy()
    x_train = x_train.iloc[:threshold, :].copy()
    
    assert x_train.shape[0] + x_val.shape[0] + x_test.shape[0] == df.shape[0] 
else:
    x_train, x_test = train_test_split(df, test_size=0.2, random_state=0)
    
    mean_values = x_train.mean()
    std_values = x_train.std()
    
    x_train, x_val = train_test_split(x_train, test_size=0.2, random_state=0)


x_train = x_train.sample(frac=1, axis=0)
x_val = x_val.sample(frac=1, axis=0)
x_test = x_test.sample(frac=1, axis=0)

def corrupt_data_interval(df, probs, augment_data=True):
    n_rows = df.shape[0]
    n_columns = df.shape[1]    
    
    if augment_data:
        data_augmentation_factor = len(probs)    
        df = np.repeat(df, data_augmentation_factor, axis=0)
        chunks = [n_rows]*len(probs)
    else:
        k = len(probs)
        size = n_rows//k
        chunks = [size]*(k-1)
        chunks.append(size+n_rows%k)
        
    month_mask = np.ones((df.shape[0], 1))
    
    mask = None
    for p, nr in zip(probs,chunks):        
        _mask = np.random.binomial(1, 1-p, size=(nr, n_columns-1))      
        if mask is None:
             mask = _mask
        else:            
            mask = np.concatenate((mask, _mask), axis=0)
    mask = np.concatenate((month_mask, mask), axis=1)
    return df, df*mask, mask   
   

def get_rmse_mean_imputation(predicted, y, std=None, mask=None):
    rmse = (predicted - y)
    if not (std is None):
        rmse = rmse*std
    if not (mask is None):
        rmse = rmse*(1-mask) #considering only corrputed columns
    rmse = np.sqrt(np.square(rmse).mean(axis=0))
    return rmse

#### MEAN IMPUTATION
def df_imputation(df, col_mean, idxs, inplace=False):
    if not inplace:
        df = df.copy()
    #idxs = np.where(df == 0.0)
    #df[idxs] = np.nan    
    #col_mean = np.nanmean(df, axis=0)
    df[idxs] = np.take(col_mean, idxs[1])
    return df    

def knn_imputation(k, x_train, x_test):
    from sklearn.neighbors import NearestNeighbors
    knn = NearestNeighbors(n_neighbors=k)
    
    val_rmse = []
    test_rmse = []    
    
    for p in probs:
        _x_test, _x_test_corrupted, _x_test_mask = corrupt_data_interval(x_test, [p], False)
        
        for _row, _mask in zip(_x_test_corrupted, _x_test_mask):
            matrix = x_train*_mask
            knn.fit(matrix)
            
            x = _row.reshape((1, _row.shape[0]))
            idxs = knn.kneighbors(x, return_distance=False)[0]
            
            mean_values = np.mean(x_train[idxs], axis=0)
            
            idxs = np.where(_mask == 0.0)[0]
            _row[idxs] = np.take(mean_values, idxs)
        _val = np.mean(get_rmse_mean_imputation(_x_test_corrupted, _x_test, mask=_mask))
        _test = 0.0
        
        val_rmse.append(_val)
        test_rmse.append(_test)
        print('RMSE (val, test) %.2f: %.4f \t %.4f' %(p, np.mean(_val), np.mean(_test)))
    return val_rmse, test_rmse

k_values = [1,3,5,7,10]
h = []
for k in k_values:
    print("K=%d"%(k))
    a, _  = knn_imputation(k, x_train.values, x_val.values)
    h.append(np.mean(a))

best_k = k_values[np.argmin(h)]
a, _  = knn_imputation(best_k, x_train.values, x_test.values)

def mean_imputation(probs, x_val, x_test, mean_values):
    
    val_rmse = []
    test_rmse = []
    
    for p in probs:
        _x_val, _x_val_corrupted, x_val_mask = corrupt_data_interval(x_val, [p], augment_data)
        _x_test, _x_test_corrupted, x_test_mask = corrupt_data_interval(x_test, [p], augment_data)
    
        val_idx = np.where(x_val_mask == 0.0)
        test_idx = np.where(x_test_mask == 0.0)
        val_imputed = df_imputation(_x_val_corrupted, mean_values, val_idx)
        test_imputed = df_imputation(_x_test_corrupted, mean_values, test_idx)
        
        _val = np.mean(get_rmse_mean_imputation(val_imputed, _x_val, mask=x_val_mask))
        _test = np.mean(get_rmse_mean_imputation(test_imputed, _x_test, mask=x_test_mask))
        print('RMSE (val, test) %.2f: %.4f \t %.4f' %(p, np.mean(_val), np.mean(_test)))
        
        val_rmse.append(_val)
        test_rmse.append(_test)
    return val_rmse, test_rmse
    # for p in probs:
    #     _x_val, _x_val_corrupted, x_val_mask = corrupt_data_interval(x_val, [p], augment_data)
    #     _x_test, _x_test_corrupted, x_test_mask = corrupt_data_interval(x_test, [p], augment_data)
    #
    #
    #     for i, name in enumerate(x_test.columns):
    #         _x_val_corrupted.loc[_x_val_corrupted[name] == 0.0, name] = mean_values[i]
    #         _x_test_corrupted.loc[_x_test_corrupted[name] == 0.0, name] = mean_values[i]
    #
    #
    #     _rmse_val = get_rmse_mean_imputation(_x_val_corrupted, x_val, mask=x_val_mask)
    #     _rmse_test = get_rmse_mean_imputation(_x_test_corrupted, x_test, mask=x_test_mask)
    #     print('RMSE (val, test) %.2f: %.4f \t %.4f' %(p, np.mean(_rmse_val), np.mean(_rmse_test)))

mean_val_rmse, mean_test_rmse = mean_imputation(probs, x_val.values, x_test.values, mean_values.values)

#%%

# standardizing data
x_train = (x_train - mean_values)/std_values
x_val = (x_val - mean_values)/std_values
x_test = (x_test - mean_values)/std_values

#converting from DataFrame to numpy array as Keras accepts numpy arrays
x_train = x_train.values
x_val = x_val.values
x_test = x_test.values


_x_train, _x_train_corrupted, _x_train_mask = corrupt_data_interval(x_train, probs, augment_data)
_x_val, _x_val_corrupted, _x_val_mask = corrupt_data_interval(x_val, probs, augment_data)






#### SVD IMPUTATION
#https://machinelearningmastery.com/singular-value-decomposition-for-machine-learning/

if False:
    idx = np.where(_x_val_corrupted[:,1:] == 0.0)
    kk = mean_imputation(_x_val_corrupted[:,1:], mean_values[1:].values, inplace=True)
    
    
    
    from scipy.linalg import svd
    
    U, s, VT = svd(kk)


#### SVD IMPUTATION

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



x_val_predicted = autoencoder.predict(_x_val_corrupted)
rmse_val = get_rmse_mean_imputation(x_val_predicted, _x_val[:,1:], std_values.values[1:], mask=_x_val_mask[:,1:])
rmse_val_full = get_rmse_mean_imputation(x_val_predicted, _x_val[:,1:], std_values.values[1:], mask=None)
print('RMSE-val: %.2f' %(np.mean(rmse_val)))
print('RMSE-val Full: %.2f' %(np.mean(rmse_val_full)))


ae_val_rmse = []
ae_test_rmse = []

for p in probs:
    _x_val, _x_val_corrupted, _x_val_mask = corrupt_data_interval(x_val, [p], augment_data)
    _x_test, _x_test_corrupted, _x_test_mask = corrupt_data_interval(x_test, [p], augment_data)
    
    _x_val_predicted = autoencoder.predict(_x_val_corrupted)
    _x_test_predicted = autoencoder.predict(_x_test_corrupted)
    
    _rmse_val = get_rmse_mean_imputation(_x_val_predicted, _x_val[:,1:], std_values.values[1:], mask=_x_val_mask[:,1:])    
    _rmse_test = get_rmse_mean_imputation(_x_test_predicted, _x_test[:,1:], std_values.values[1:], mask=_x_test_mask[:,1:])    
    
    _val = np.mean(_rmse_val)
    _test = np.mean(_rmse_test) 
    print('RMSE (val, test) %.2f: %.4f \t %.4f' %(p, _val, _test))
    
    ae_val_rmse.append(_val)
    ae_test_rmse.append(_test)


test_improv = 100*(np.array(ae_test_rmse) / np.array(mean_test_rmse))
val_improv  = 100*(np.array(ae_val_rmse) / np.array(mean_val_rmse))
    

test_improv = np.round(test_improv, decimals=2)
val_improv = np.round(val_improv, decimals=2)

for p, v, t in zip(probs, val_improv, test_improv):
    print('Mean/AE (val, test) %.2f: %.2f \t %.2f' %(p, v, t))

#rmse_test_full = get_rmse(x_test_predicted, x_test, std_values.values, mask=None)

#print('RMSE-test Full: %.2f' %(np.mean(rmse_test_full)))


if False:
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


