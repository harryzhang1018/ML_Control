from keras.models import Sequential
from keras.layers import Dense
from numpy import genfromtxt
import numpy as np

train_set_1 = genfromtxt('./data/mpc_sin_train1.csv', delimiter=',')
train_set_2 = genfromtxt('./data/mpc_sin_train2.csv', delimiter=',')
train_set_3 = genfromtxt('./data/mpc_sin_train3.csv', delimiter=',')
##combining training data into one big matrix
train_set = np.concatenate((train_set_1,train_set_2,train_set_3),axis=0)
#train_set = np.unique(train_set,axis=0)

error_state = train_set[:,6:10] # error states--inputs
ctrl_output = train_set[:,4:6] # control outputs

#model = Sequential()
#model.add(Dense(8, input_dim=4, activation='relu'))
#model.add(Dense(16, activation='relu'))
#model.add(Dense(2, activation='linear'))
#model.compile(optimizer='adam', loss='mse', metrics=['mae'])
#history = model.fit(error_state, ctrl_output, epochs=100, batch_size=32, validation_data=(error_state, ctrl_output))
#loss, mae = model.evaluate(error_state, ctrl_output)

#print('Test loss:', loss)
#print('Test MAE:', mae)
#model.save('keras_m1.h5')


## for training EKF

ekf_t1 = genfromtxt('./data/ekf_sin_t1.csv',delimiter=',')
ekf_t2 = genfromtxt('./data/ekf_sin_t2.csv',delimiter=',')
ekf_t3 = genfromtxt('./data/ekf_sin_t3.csv',delimiter=',')

ekf_train_set = np.concatenate( (ekf_t1, ekf_t2, ekf_t3),axis = 0 )

train_inputs = ekf_train_set[:, [0, 1, -2, -1]]
train_inputs = train_inputs.reshape((-1,4))

train_outputs = ekf_train_set[:,[2,3]]

ekf_model = Sequential()
ekf_model.add(Dense(8, input_dim=4, activation='relu'))
ekf_model.add(Dense(16, activation='relu'))
ekf_model.add(Dense(2, activation='linear'))
ekf_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
ekf_history = ekf_model.fit(train_inputs, train_outputs, epochs=200, batch_size=16, validation_data=(train_inputs, train_outputs))
loss, mae = ekf_model.evaluate(train_inputs, train_outputs)

print('Test loss:', loss)
print('Test MAE:', mae)
ekf_model.save('keras_ekf_m1.h5')



















