#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np


def RCNN(signal_input_shape, one_hot_input_shape, dropout=False):
    training = True if dropout else None

    image_input = tf.keras.Input(signal_input_shape)
    one_hot = tf.keras.Input((one_hot_input_shape,))
    one_hot_repeat = tf.keras.layers.RepeatVector(image_input.shape[1])(one_hot)

    # Noise added
    #noise = tf.keras.layers.GaussianNoise(np.std(original_train_set))(image_input)
    conv1 = tf.keras.layers.Conv2D(64, (1, 3), activation='relu')(image_input)
    batchnorm1 = tf.keras.layers.BatchNormalization()(conv1)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(1, 2))(batchnorm1)
    dropout1 = tf.keras.layers.Dropout(0.5)(pool1, training=training)
    conv2 = tf.keras.layers.Conv2D(32, (1, 2), activation='relu')(dropout1)
    batchnorm2 = tf.keras.layers.BatchNormalization()(conv2)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(1, 2))(batchnorm2)
    dropout2 = tf.keras.layers.Dropout(0.5)(pool2, training=training)
    conv3 = tf.keras.layers.Conv2D(32, (1, 3), activation='relu')(dropout2)
    batchnorm3 = tf.keras.layers.BatchNormalization()(conv3)

    reshape = tf.keras.layers.Reshape((image_input.shape[1], conv3.shape[-1]))(batchnorm3)
    concat = tf.keras.layers.Concatenate()([reshape, one_hot_repeat])
    rnn = tf.keras.layers.GRU(8, dropout=0.5)(concat, training=training)
    output = tf.keras.layers.Dense(1, name='output')(rnn)

    model = tf.keras.Model([image_input, one_hot], [output])

    return model


def RCNN_truck(signal_input_shape, dropout=False):
    training = True if dropout else None

    image_input = tf.keras.Input(signal_input_shape)

    # Conv layers
    conv1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(16, (3, 3), strides=(1, 1), padding='same', activation='relu'))(image_input)
    batch1 = tf.keras.layers.BatchNormalization()(conv1)
    conv2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu'))(batch1)
    batch2 = tf.keras.layers.BatchNormalization()(conv2)
    pool1 = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(3, 3)))(batch2)

    conv3 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(16, (3, 3), strides=(1, 1), padding='same', activation='relu'))(pool1)
    batch3 = tf.keras.layers.BatchNormalization()(conv3)
    conv4 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu'))(batch3)
    batch4 = tf.keras.layers.BatchNormalization()(conv4)
    pool2 = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(3, 3)))(batch4)

    conv5 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu'))(pool2)
    batch5 = tf.keras.layers.BatchNormalization()(conv5)
    conv6 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu'))(batch5)
    batch6 = tf.keras.layers.BatchNormalization()(conv6)
    reshape = tf.keras.layers.Reshape(
        (batch6.shape[1], batch6.shape[2], batch6.shape[3]*batch6.shape[4])
    )(batch6)

    #RNN
    rnn = tf.keras.layers.TimeDistributed(
        tf.keras.layers.GRU(5,
        dropout=0.5)
    )(reshape, training=training)

    # Flatten
    flattened = tf.keras.layers.Flatten()(rnn)

    # Output
    dense1 = tf.keras.layers.Dense(5, name='output')(flattened)

    model = tf.keras.Model(inputs=[image_input], outputs=[dense1])

    return model

def CNN_truck(signal_input_shape, output_bias=None, classification=False, dropout=False):
    training = True if dropout else None

    image_input = tf.keras.Input(signal_input_shape)

    # Conv layers
    conv1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(16, (3, 3), strides=(1, 1), padding='same', activation='relu'))(image_input)
    batch1 = tf.keras.layers.BatchNormalization()(conv1)
    conv2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu'))(batch1)
    batch2 = tf.keras.layers.BatchNormalization()(conv2)
    pool1 = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))(batch2)

    conv3 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(16, (3, 3), strides=(1, 1), padding='same', activation='relu'))(pool1)
    batch3 = tf.keras.layers.BatchNormalization()(conv3)
    conv4 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu'))(batch3)
    batch4 = tf.keras.layers.BatchNormalization()(conv4)
    pool2 = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))(batch4)

    conv5 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu'))(pool2)
    batch5 = tf.keras.layers.BatchNormalization()(conv5)
    conv6 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu'))(batch5)
    batch6 = tf.keras.layers.BatchNormalization()(conv6)

    # Flatten
    flattened = tf.keras.layers.Flatten()(batch6)

    # Output
    if classification:
        assert output_bias is not None, 'output_bias is undefined. Please specify value.'
        output_bias_ = tf.keras.initializers.Constant(output_bias)        
        dense1 = tf.keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias_, name='output')(flattened)
    else:
        dense1 = tf.keras.layers.Dense(1, activation='relu', name='output')(flattened)
        # dense1 = tf.keras.layers.Dense(5, activation='relu', name='output')(flattened)

    model = tf.keras.Model(inputs=[image_input], outputs=[dense1])

    return model


def load_1DCNNTemp(signal_input_shape, output_bias=None, classification=False, dropout=False):
    training = True if dropout else None

    # input size
    print(signal_input_shape)
    peaks_input = tf.keras.Input(signal_input_shape)
    #spreadings_input = Input((n_peaks - 1,))
    # layer 1
    conv1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(filters=32, kernel_size=3, strides=10, activation='linear', use_bias=False))(peaks_input)
    bn1 = tf.keras.layers.BatchNormalization()(conv1)
    act1 = tf.keras.layers.Activation('relu')(bn1)
    pool1 = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling1D(8))(act1)
    dp1 = tf.keras.layers.Dropout(0.1)(pool1, training=training)
    # layer 2
    conv2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(filters=16, kernel_size=3, activation='linear', use_bias=False))(dp1)
    bn2 = tf.keras.layers.BatchNormalization()(conv2)
    act2 = tf.keras.layers.Activation('relu')(bn2)
    dp2 = tf.keras.layers.Dropout(0.1)(act2, training=training)
    flat = tf.keras.layers.Flatten()(dp2)
    # concat output of convolution and spreading data
    #concat = concatenate([flat, spreadings_input])
    concat = flat
    # fully connected layer
    dense = tf.keras.layers.Dense(units=10, activation='linear', use_bias=False)(concat)
    bn = tf.keras.layers.BatchNormalization()(dense)
    act = tf.keras.layers.Activation('relu')(bn)
    # output layer
    if classification:
        assert output_bias is not None, 'output_bias is undefined. Please specify value.'
        output_bias_ = tf.keras.initializers.Constant(output_bias)
        layer_output = tf.keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias_, name='output')(act)
    else:
        layer_output = tf.keras.layers.Dense(1, name='output')(act)
    # model
    model = tf.keras.Model(inputs=[peaks_input], outputs=[layer_output])

    return model    

