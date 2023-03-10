# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 15:35:38 2023

@author: asus
"""
#import tensorflow as tf
#from sklearn.model_selection import StratifiedKFold
#from keras.layers import GaussianNoise

#from keras import optimizers


import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.model_selection import train_test_split
from bayes_opt import BayesianOptimization
import os
from sklearn.preprocessing import MinMaxScaler
import tensorflow 



path = r'C:\Users\asus\Desktop\Semester 2\Intelligent Analytics\HW2 - Multi Layer Perceptron\Font Recognition'


### Problem 1 using Bays




path = r'C:\Users\asus\Desktop\Semester 2\Intelligent Analytics\HW2 - Multi Layer Perceptron\Font Recognition'

# Data Import
data = pd.read_excel(os.path.join(path , 'dataset.xlsx') , sheet_name= "Font Recognition - Train 0")
data_test = pd.read_excel(os.path.join(path , 'dataset.xlsx') , sheet_name= "Font Recognition - Test")

inputs  = data.iloc[:,0:14] 
outputs  = data.iloc[:,14:40]   

inputs_test = data_test.iloc[:, 0:14] 
output_test = data_test.iloc[:,14:40]


X_train, X_val, y_train, y_val = train_test_split(inputs, outputs, test_size=1/3, stratify=outputs, random_state=50)

#Normalizing the data
scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(inputs_test)

y_test  = output_test
#X_test_norm = scaler.transform(inputs_test)


# Model Fitting
num_classes = outputs.shape[1]
num_features = inputs.shape[1]

earlystopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')

# Define the MLP model

# Define the objective function dropout_rate,, dropout_layer_idx
def mlp_objective( num_layers, neurons_per_layer_0, neurons_per_layer_1, 
                  neurons_per_layer_2, neurons_per_layer_3, neurons_per_layer_4):
    # Build the model with the given hyperparameters
    model = Sequential()
    model.add(Dense(int(neurons_per_layer_0), activation='relu', input_shape=(X_train.shape[1],)))
    for i in range(1, int(num_layers)):
        # if i == int(dropout_layer_idx):
        #     model.add(Dropout(rate=dropout_rate))
        model.add(Dense(int(eval(f'neurons_per_layer_{i}')), activation='relu' , kernel_regularizer=tensorflow.keras.regularizers.l2(0.01)  ))
    model.add(Dense(26, activation='softmax'))
    #optimizer = Adam(lr=0.001)
    model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])

    # Train the model and evaluate on the validation set
    history = model.fit(X_train, y_train, epochs=400, batch_size=52, validation_data=(X_val, y_val), verbose=0 , callbacks =[earlystopping])
    val_acc = np.max(history.history['val_accuracy'])

    return val_acc

# Define the search space for the hyperparameters
pbounds = {#'dropout_rate': (0, 0.5),
           'num_layers': (1, 5),
           'neurons_per_layer_0': (16, 256),
           'neurons_per_layer_1': (16, 256),
           'neurons_per_layer_2': (16, 256),
           'neurons_per_layer_3': (16, 256),
           'neurons_per_layer_4': (16, 256),
           #'dropout_layer_idx': (-1, 4)
           }

# Instantiate the optimizer
optimizer = BayesianOptimization(f=mlp_objective, pbounds=pbounds, random_state=42)

# Run the optimization
optimizer.maximize(init_points=5, n_iter=100, acq='ei')

# Get the best hyperparameters
best_params = optimizer.max['params']


# Build the final model with the best hyperparameters
model = Sequential()
model.add(Dense(int(best_params['neurons_per_layer_0']), activation='relu', input_shape=(X_train.shape[1],)))
for i in range(1, int(best_params['num_layers'])):
 #   if i == int(best_params['dropout_layer_idx']):
 #       model.add(Dropout(rate=best_params['dropout_rate']))
    model.add(Dense(int(best_params[f'neurons_per_layer_{i}']), activation='relu' , kernel_regularizer=tensorflow.keras.regularizers.l2(0.01)  ))
model.add(Dense(26, activation='sigmoid'))
#optimizer = Adam(lr=0.001)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the final model
history = model.fit(X_train, y_train, epochs=300, batch_size=52 , verbose = 1 ,  callbacks =[earlystopping])


test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)


