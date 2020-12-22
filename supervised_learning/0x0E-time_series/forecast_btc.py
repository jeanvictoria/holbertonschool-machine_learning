#!/usr/bin/python3
"""
Forcast
"""
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
create_timeSteps = __import__('timestep_plots').create_timeSteps
multivariate_data = __import__('variate.py').multivariate_data
plot_trainHistory = __import__('timestep_plots').plot_trainHistory
preprocess = __import__('preprocess_data').preprocess_data
show_plot = __import__('timestep_plots').show_plot


mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False


pastHistory = 24
futureTarget = 0


TRAIN_SPLIT = 1300000
STEP = 1

# dataset = multivariate_data(TRAIN_SPLIT)
dataset = preprocess("coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv")

x_train_single, y_train_single = multivariate_data(dataset, dataset[:, 2],
                                                   0, TRAIN_SPLIT,
                                                   pastHistory,
                                                   futureTarget, STEP,
                                                   single_step=True)
x_val_single, y_val_single = multivariate_data(dataset, dataset[:, 2],
                                               TRAIN_SPLIT, None,
                                               pastHistory,
                                               futureTarget, STEP,
                                               single_step=True)
# print('Single Window of Past History : {}'.format(x_train_single[0].shape))

BUFFER_SIZE = 10000
BATCH_SIZE = 256

train_data_single = tf.data.Dataset.from_tensor_slices((x_train_single,
                                                        y_train_single))
train_data_single = train_data_single.cache().shuffle(BUFFER_SIZE).batch(
    BATCH_SIZE).repeat()

val_data_single = tf.data.Dataset.from_tensor_slices((x_val_single,
                                                      y_val_single))
val_data_single = val_data_single.batch(BATCH_SIZE).repeat()

single_step_model = tf.keras.models.Sequential()
single_step_model.add(tf.keras.layers.LSTM
                      (24, input_shape=x_train_single.shape[-2:]))
single_step_model.add(tf.keras.layers.Dense(1))
# generate model summary
single_step_model.summary()

single_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(),
                          loss='mse')

early_stopping = EarlyStopping(monitor='val_loss',
                               patience=4,
                               restore_best_weights=True)

EPOCHS = 30
EVAL_INTERVAL = 500
single_stepHistory = single_step_model.fit(train_data_single,
                                           epochs=EPOCHS,
                                           steps_per_epoch=EVAL_INTERVAL,
                                           validation_data=val_data_single,
                                           callbacks=[early_stopping],
                                           validation_steps=50)


plot_trainHistory(single_stepHistory,
                  'Single-Step Training and  Validation Loss')

for x, y in val_data_single.take(10):
    plot = show_plot([x[0][:, 1].numpy(), y[0].numpy(),
                      single_step_model.predict(x)[0]], 1,
                     'Single-Step Prediction')
plot.show()
