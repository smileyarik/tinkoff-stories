import os
import sys
import csv

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from keras.layers import Input, Embedding, LSTM, Dense, Dot, Reshape, Flatten, Concatenate, Dropout
from keras.models import Model
import numpy as np

import tensorflow as tf
from sklearn.metrics import roc_auc_score
from collections import defaultdict

from learn_nn2 import tf_score, load_users_and_items

if __name__ == '__main__':
    EMBED_SIZE=8
    #MAX_USER_ID=1103495
    #MAX_ITEM_ID=1100028
    MAX_USER_ID=35000
    MAX_ITEM_ID=1000

    classes = {'dislike' : 0, 'skip' : 1, 'view' : 2, 'like' : 3}

    np.random.seed(3)

    user_train = []
    item_train = []
    y_train = []

    user_test = []
    item_test = []
    y_test = []

    user_map, item_map = load_users_and_items(sys.argv[3], sys.argv[4])
    print(len(user_map), len(item_map))


    for line in open(sys.argv[1]):
        line = line.strip()
        if len(line) == 0: continue # stupid windows
        ff = line.split(',')
        user = int(ff[0])
        item = int(ff[1])
        user_train.append([user_map[user].idx])
        item_train.append([item_map[item]])
        y = [0, 0, 0, 0]
        y[classes[ff[3]]] = 1
        y_train.append(y)

    for line in open(sys.argv[2]):
        line = line.strip()
        if len(line) == 0: continue # stupid windows
        ff = line.split(',')
        user = int(ff[0])
        item = int(ff[1])
        user_test.append([user_map[user].idx])
        item_test.append([item_map[item]])
        y = [0, 0, 0, 0]
        y[classes[ff[3]]] = 1
        y_test.append(y)

    print(len(y_train), sum(sum(x) for x in y_train))
    print(len(y_test), sum(sum(x) for x in y_test))
    print(np.array(user_train).shape)
    print(np.array(y_train).shape)


    user_input = Input(shape=(1,), dtype='int32')
    user_embed = Dropout(1.0)(Embedding(len(user_map) + 1, EMBED_SIZE, input_length=1)(user_input))
    user_embed_2 = Flatten()(user_embed)
    user_bias = Embedding(len(user_map) + 1, 2, input_length=1)(user_input)
    user_bias_2 = Flatten()(user_bias)


    print(user_embed.shape)
    print(user_embed_2.shape)

    item_input = Input(shape=(1,), dtype='int32')
    item_embed = Dropout(1.0)(Embedding(len(item_map) + 1, EMBED_SIZE, input_length=1)(item_input))
    item_embed_2 = Flatten()(item_embed)
    item_bias = Embedding(len(item_map) + 1, 2, input_length=1)(item_input)
    item_bias_2 = Flatten()(item_bias)

    print(item_embed.shape)
    print(item_embed_2.shape)

    dot_layer = Dot(axes=1)([user_embed_2, item_embed_2])
    print(dot_layer.shape)
    cat_layer = Concatenate(axis=1)([user_embed_2, item_embed_2, user_bias_2, item_bias_2, dot_layer])
    print(cat_layer.shape)

    relu_layer =  Dropout(0.2)(Dense(20, activation='relu')(cat_layer))
    dense_layer = Dense(4, activation='softmax')(relu_layer)
    #print dense_layer.shape

    #relu_layer_2 =  Dropout(1.0)(Dense(10, activation='relu')(dot_layer))
    dense_layer_2 = Dense(4, activation='softmax')(relu_layer)

    #dot_model = Model(inputs=[user_input, item_input], outputs=dense_layer_2)
    #dot_model.compile(optimizer='adam',
    #              loss='categorical_crossentropy',
    #              metrics=['accuracy', tf_score])

    #dot_model.fit([np.array(user_train), np.array(item_train)],
    #    np.array(y_train),
    #    batch_size=1024,
    #    nb_epoch=5,
    #    validation_data=([np.array(user_test), np.array(item_test)], np.array(y_test)),
        #validation_split=0.2,
    #    verbose=1)
    #item_embed.trainable = False
    #user_embed.trainable = False

    model = Model(inputs=[user_input, item_input], outputs=dense_layer_2)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy', tf_score])

    model.fit([np.array(user_train), np.array(item_train)],
        np.array(y_train),
        batch_size=1024,
        nb_epoch=10,
        validation_data=([np.array(user_test), np.array(item_test)], np.array(y_test)),
        #validation_split=0.2,
        verbose=2)

    model.save(sys.argv[5])
