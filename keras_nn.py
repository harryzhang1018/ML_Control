from keras_core.models import Sequential
from keras_core.layers import Dense
from sklearn.utils import shuffle
from numpy import genfromtxt
import numpy as np
train_set_1 = genfromtxt('./data/circle_MPC_trainingData/circle_r2ccw1.csv', delimiter = ',')
train_set_2 = genfromtxt('./data/circle_MPC_trainingData/circle_r2ccw2.csv', delimiter = ',')
train_set_3 = genfromtxt('./data/circle_MPC_trainingData/circle_r2ccw3.csv', delimiter = ',')
train_set_4 = genfromtxt('./data/circle_MPC_trainingData/circle_r2cw1.csv', delimiter = ',')
train_set_5 = genfromtxt('./data/circle_MPC_trainingData/circle_r2cw2.csv', delimiter = ',')
#train_set_6 = genfromtxt('./data/circle_MPC_trainingData/circle_r2cw3.csv', delimiter = ',')
train_set_7 = genfromtxt('./data/circle_MPC_trainingData/circle_r5ccw1.csv', delimiter = ',')
train_set_8 = genfromtxt('./data/circle_MPC_trainingData/circle_r5ccw2.csv', delimiter = ',')
train_set_9 = genfromtxt('./data/circle_MPC_trainingData/circle_r5ccw3.csv', delimiter = ',')
train_set_10 = genfromtxt('./data/circle_MPC_trainingData/circle_r5cw1.csv', delimiter = ',')
train_set_11 = genfromtxt('./data/circle_MPC_trainingData/circle_r5cw2.csv', delimiter = ',')
#train_set_12 = genfromtxt('./data/circle_MPC_trainingData/circle_r5cw3.csv', delimiter = ',')
train_set_13 = genfromtxt('./data/circle_MPC_trainingData/circle_r25ccw123.csv', delimiter = ',')
train_set_14 = genfromtxt('./data/circle_MPC_trainingData/circle_r25cw123.csv', delimiter = ',')
train_set_15 = genfromtxt('./data/circle_MPC_trainingData/circle_rinf1.csv', delimiter = ',')
train_set_16 = genfromtxt('./data/circle_MPC_trainingData/circle_rinf2.csv', delimiter = ',')
train_set_17 = genfromtxt('./data/circle_MPC_trainingData/circle_rinf3.csv', delimiter = ',')
# train_set_1 = genfromtxt('./data/sin_sim_testing_1.csv', delimiter=',')
# train_set_2 = genfromtxt('./data/sin_sim_testing_2.csv', delimiter=',')
# train_set_3 = genfromtxt('./data/sin_sim_testing_3.csv', delimiter=',')
# train_set_4 = genfromtxt('./data/sin_sim_testing_4.csv', delimiter=',')
# train_set_5 = genfromtxt('./data/sin_sim_testing_5.csv', delimiter=',')
# train_set_5 = genfromtxt('./data/sin_sim_testing_5.csv', delimiter=',')
# train_set_6 = genfromtxt('./data/circle_sim_testing_1.csv', delimiter=',')
# train_set_7 = genfromtxt('./data/circle_sim_testing_2.csv', delimiter=',')
# train_set_8 = genfromtxt('./data/circle_sim_testing_3.csv', delimiter=',')
# train_set_9 = genfromtxt('./data/circle_sim_testing_4.csv', delimiter=',')
# train_set_10 = genfromtxt('./data/circle_sim_testing_5.csv', delimiter=',')
##combining training data into one big matrix
train_set = np.concatenate((train_set_1,train_set_2,train_set_3,train_set_4,train_set_5,train_set_7,train_set_8,train_set_9,train_set_10,train_set_11,train_set_13,train_set_14,train_set_15,train_set_16,train_set_17),axis=0)#,train_set_6,train_set_7,train_set_8,train_set_9,train_set_10),axis=0)
train_set = shuffle(train_set)


# train_set = genfromtxt('./data.csv', delimiter=',')
error_state = train_set[:,2:6] #error states--inputs
ctrl_output = train_set[:,6:8] # control outputs

model = Sequential()
model.add(Dense(8, input_dim=4, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(2, activation='linear'))
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
history = model.fit(error_state, ctrl_output, epochs=100, batch_size=32, validation_data=(error_state, ctrl_output), shuffle = True)
loss, mae = model.evaluate(error_state, ctrl_output)

print('Test loss:', loss)
print('Test MAE:', mae)
model.save('keras_ml_learnMC.keras')


# ## for training EKF

# ekf_t1 = genfromtxt('./data/ekf_sin_t1.csv',delimiter=',')
# ekf_t2 = genfromtxt('./data/ekf_sin_t2.csv',delimiter=',')
# ekf_t3 = genfromtxt('./data/ekf_sin_t3.csv',delimiter=',')

# ekf_train_set = np.concatenate( (ekf_t1, ekf_t2, ekf_t3),axis = 0 )

# train_inputs = ekf_train_set[:, [0, 1, -2, -1]]
# train_inputs = train_inputs.reshape((-1,4))

# train_outputs = ekf_train_set[:,[2,3]]

# ekf_model = Sequential()
# ekf_model.add(Dense(8, input_dim=4, activation='relu'))
# ekf_model.add(Dense(16, activation='relu'))
# ekf_model.add(Dense(2, activation='linear'))
# ekf_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
# ekf_history = ekf_model.fit(train_inputs, train_outputs, epochs=200, batch_size=16, validation_data=(train_inputs, train_outputs))
# loss, mae = ekf_model.evaluate(train_inputs, train_outputs)

# print('Test loss:', loss)
# print('Test MAE:', mae)
# ekf_model.save('keras_ekf_m1.h5')





























