from keras_core.models import Model
from keras_core.layers import Dense, Input, concatenate
from sklearn.utils import shuffle
from numpy import genfromtxt
import numpy as np
import keras_core.backend as K
#control is throttle, steering
def throttle_loss(y_true, y_pred):
    control1_true = y_true[:,0]
    control2_true = y_true[:,1]
    control1_pred = y_pred[:,0]
    control2_pred = y_pred[:,1]
    #throttle_loss
    loss_control1 = 0.8*(mse(control1_true, control1_pred))
    #steering_loss
    loss_control2 = 0.2*(mse(control2_true,control2_pred))
    total_loss = loss_control1+loss_control2
    return total_loss

def steering_loss(y_true, y_pred):
    control1_true = y_true[:,0]
    control2_true = y_true[:,1]
    control1_pred = y_pred[:,0]
    control2_pred = y_pred[:,1]
    #throttle_loss
    loss_control1 = 0.2*mse(control1_true,control1_pred)
    #steering_loss
    loss_control2 = 0.8*mse(control2_true,control2_pred)
    total_loss = loss_control1+loss_control2
    return total_loss

def mse(y_true, y_pred):
    final = y_true
    for i in range(0,y_true.size):
        final[i] = (y_true[i]-y_pred[i])**2
    return final
#setting up training sets
train_set_1 = genfromtxt('./data/circle_MPC_trainingData/circle_r2ccw1.csv', delimiter = ',')
train_set_2 = genfromtxt('./data/circle_MPC_trainingData/circle_r2ccw2.csv', delimiter = ',')
train_set_3 = genfromtxt('./data/circle_MPC_trainingData/circle_r2ccw3.csv', delimiter = ',')
train_set_4 = genfromtxt('./data/circle_MPC_trainingData/circle_r2cw1.csv', delimiter = ',')
train_set_5 = genfromtxt('./data/circle_MPC_trainingData/circle_r2cw2.csv', delimiter = ',')
train_set_7 = genfromtxt('./data/circle_MPC_trainingData/circle_r5ccw1.csv', delimiter = ',')
train_set_8 = genfromtxt('./data/circle_MPC_trainingData/circle_r5ccw2.csv', delimiter = ',')
train_set_9 = genfromtxt('./data/circle_MPC_trainingData/circle_r5ccw3.csv', delimiter = ',')
train_set_10 = genfromtxt('./data/circle_MPC_trainingData/circle_r5cw1.csv', delimiter = ',')
train_set_11 = genfromtxt('./data/circle_MPC_trainingData/circle_r5cw2.csv', delimiter = ',')
train_set_13 = genfromtxt('./data/circle_MPC_trainingData/circle_r25ccw123.csv', delimiter = ',')
train_set_14 = genfromtxt('./data/circle_MPC_trainingData/circle_r25cw123.csv', delimiter = ',')
train_set_15 = genfromtxt('./data/circle_MPC_trainingData/circle_rinf1.csv', delimiter = ',')
train_set_16 = genfromtxt('./data/circle_MPC_trainingData/circle_rinf2.csv', delimiter = ',')
train_set_17 = genfromtxt('./data/circle_MPC_trainingData/circle_rinf3.csv', delimiter = ',')

train_set = np.concatenate((train_set_1,train_set_2,train_set_3,train_set_4,train_set_5,train_set_7,train_set_8,train_set_9,train_set_10,train_set_11,train_set_13,train_set_14,train_set_15,train_set_16,train_set_17),axis=0)
train_set = shuffle(train_set)

error_state = train_set[:,2:6]
ctrl_output = train_set[:,6:8]

#setting up model
input_layer = Input(shape=(4,))
#branch 1: throttle
branch11 = Dense(8, activation='relu', name='branch11')(input_layer)
branch12 = Dense(16,activation='relu', name='branch12')(branch11)
branch13 = Dense(2,activation='linear', name='branch13')(branch12)

#branch 2: steering
branch21 = Dense(8, activation='relu', trainable = False, name='branch21')(input_layer)
branch22 = Dense(16,activation='relu', trainable = False, name='branch22')(branch21)
branch23 = Dense(2,activation='linear', trainable = False, name='branch23')(branch22)

combined_output = concatenate([branch13, branch23])

combined_layer1 = Dense(8, activation='relu', trainable=False, name='combined1')(combined_output)
combined_layer2 = Dense(16, activation='relu', trainable=False, name='combined2')(combined_layer1)
final_output1 = Dense(1, activation='linear', trainable=False, name='final1')(combined_layer2)
final_output2 = Dense(1, activation='linear', trainable=False, name='final2')(combined_layer2)


model = Model(inputs = input_layer, outputs = [final_output1,final_output2])

model.compile(optimizer='adam', loss={'final1': 'mse', 'final2': 'mse'}, loss_weights={'final1': 0.8, 'final2': 0.2}, metrics=['mae', 'mae'])

#trianing for throttle

history1 = model.fit(error_state, ctrl_output, epochs=100, batch_size=32, validation_data=(error_state, ctrl_output), shuffle=True)

loss, mae = model.evaluate(error_state, ctrl_output)
print('Test loss:', loss)
print('Test MAE:', mae)

#training for steering
model.get_layer('branch11').trainable = False
model.get_layer('branch12').trainable = False
model.get_layer('branch13').trainable = False
model.get_layer('branch21').trainable = True
model.get_layer('branch22').trainable = True
model.get_layer('branch23').trainable = True

model.compile(optimizer='adam', loss={'final1': 'mse', 'final2': 'mse'}, loss_weights={'final1': 0.2, 'final2': 0.8}, metrics=['mae', 'mae'])
history2 = model.fit(error_state, ctrl_output, epochs=100, batch_size=32, validation_data=(error_state, ctrl_output), shuffle=True)
loss, mae = model.evaluate(error_state, ctrl_output)
print('Test loss:', loss)
print('Test MAE:', mae)

#training combined

model.get_layer('branch21').trainable = False
model.get_layer('branch22').trainable = False
model.get_layer('branch23').trainable = False
model.get_layer('combined1').trainable = True
model.get_layer('combined2').trainable = True
model.get_layer('final1').trainable = True
model.get_layer('final2').trainable = True


model.compile(optimizer='adam', loss={'final1': 'mse', 'final2': 'mse'}, metrics=['mae', 'mae'])

history = model.fit(error_state, ctrl_output, epochs=100, batch_size=32, validation_data=(error_state, ctrl_output), shuffle=True)
loss, mae = model.evaluate(error_state, ctrl_output)
print('Test loss:', loss)
print('Test MAE:', mae)

#save the mode

#model.save('keras_functional.keras')
