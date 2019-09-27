1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
48
49
50
51
52
53
54
55
56
57
58
59
60
61
62
63
64
65
66
67
68
69
70
71
72
73
74
75
76
77
78
79
80
81
82
83
84
85
86
87
88
89
90
91
92
93
94
95
96
97
98
99
100
101
102
103
104
105
106
107
108
109
110
111
112
113
114
115
116
117
118
119
120
121
122
123
124
125
126
127
128
129
130
131
132
133
134
135
136
137
138
139
140
141
142
143
144
145
146
147
148
149
150
151
152
153
154
155
156
157
158
159
160
161
162
163
164
165
166
167
168
169
170
171
172
173
174
175
176
177
178
179
180
181
182
183
184
185
186
187
188
189
190
191
192
193
194
195
196
197
198
199
200
201
202
203
204
205
206
207
208
209
210
211
212
213
214
215
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 14:43:17 2019
 
@author: ramon
"""
 
import numpy as np
np.random.seed(0)
 
import matplotlib.pyplot as plt 
import pandas as pd
 
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
     
#from sklearn import preprocessing
#preprocessing.scale(x_train, copy=False)
#preprocessing.scale(x_test, copy=False)
 
#corrupting the data
 
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
 
#x_train_corrupted, x_train_mask = corrupt_data(x_train, corruption_prob)
#x_val_corrupted, x_val_mask = corrupt_data(x_val, corruption_prob)
#x_test_corrupted, x_test_mask = corrupt_data(x_test, corruption_prob)
 
x_train, x_train_corrupted, x_train_mask = corrupt_data_interval(x_train, probs)
x_val, x_val_corrupted, x_val_mask = corrupt_data_interval(x_val, probs)
 
 
 
#creating autoencoder
#input_ts = Input(shape=(input_dim,), dtype='float32')
#encoded = Dense(encoding_dim, activation='relu', #activity_regularizer=regularizers.l1(l1_reg))(input_ts)
#decoded = Dense(output_dim)(encoded)
 
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
 
autoencoder_train = autoencoder.fit(x_train_corrupted, x_train[:,1:], epochs=epochs, batch_size=batch_size, shuffle=True, verbose=1, validation_data=(x_val_corrupted, x_val[:,1:]), callbacks=[es])
#autoencoder_train = autoencoder.fit(x_train, x_train, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=1)
 
 
 
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
 
 
x_val_predicted = autoencoder.predict(x_val_corrupted)
rmse_val = get_rmse(x_val_predicted, x_val, std_values.values, mask=x_val_mask)
rmse_val_full = get_rmse(x_val_predicted, x_val, std_values.values, mask=None)
print('RMSE-val: %.2f' %(np.mean(rmse_val)))
print('RMSE-val Full: %.2f' %(np.mean(rmse_val_full)))
 
 
for p in probs:
    x_val, x_val_corrupted, x_val_mask = corrupt_data_interval(x_val, [p])
    x_test, x_test_corrupted, x_test_mask = corrupt_data_interval(x_test, [p])
     
    x_val_predicted = autoencoder.predict(x_val_corrupted)
    x_test_predicted = autoencoder.predict(x_test_corrupted)
     
    _rmse_test = get_rmse(x_test_predicted, x_test, std_values.values,           mask=x_test_mask)    
    _rmse_val = get_rmse(x_val_predicted, x_val, std_values.values,           mask=x_val_mask)    
    print('RMSE (val, test) %.2f: %.4f \t %.4f' %(p, np.mean(_rmse_val),  np.mean(_rmse_test)))
 
     
#rmse_test_full = get_rmse(x_test_predicted, x_test, std_values.values, mask=None)
 
#print('RMSE-test Full: %.2f' %(np.mean(rmse_test_full)))
 
 
if True:
#ploting rmse for each variable    
    objects = np.arange(len(rmse_val))
    y_pos = np.arange(len(objects))
    plt.bar(y_pos, rmse_test, align='center', alpha=0.5)
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