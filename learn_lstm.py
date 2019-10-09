import os
import sys
import csv

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from keras.utils import plot_model
from keras.layers import Input, Embedding, LSTM, Dense, Dot, Reshape, Flatten, Concatenate, Dropout, Lambda, CuDNNGRU
from keras.models import Model
from keras.preprocessing import sequence
import numpy as np
import keras.backend as K
from random import shuffle

import tensorflow as tf
from sklearn.metrics import roc_auc_score
from collections import defaultdict

def load_users_and_items(user_file, item_file):
    user_map = {}
    i = 0
    with open(user_file) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader, None)
        for row in reader:
            user_id = int(row[0])
            user_map[user_id] = i
            i += 1

    item_map = {}
    i = 0
    with open(item_file) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            item_id = int(row[0])
            item_map[item_id] = i
            i += 1
    return user_map, item_map



def auroc(y_true, y_pred):
    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)

def score(y_true, y_pred):
    weight = {0 : -10, 1 : -0.1, 2 : 0.1, 3 : 0.5}
    #print y_true[0], y_pred[0]
    res = 0.
    for i in xrange(0,len(y_true)):
        s = 0.
        r = 0.
        for j in xrange(0,4):
            s += y_pred[i][j] * weight[j]
            r += y_true[i][j] * weight[j]
        if s > 0:
            res += r
        else:
            res -= r
    #print res, len(y_true), float(res) / len(y_true)
    return float(res) / len(y_true)

def tf_score(y_true, y_pred):
    print len(y_true)
    print len(y_true[0])
    print len(y_true[0][0])

    return tf.double(0.)
    return tf.py_func(score, (y_true, y_pred), tf.double)

def load_set(path):
    user_history = defaultdict(list)

    left_train = []
    left_x_train = []
    right_train = []
    y_train = []
    w_train = []

    for line in open(path):
        ff = line.strip().split(',')
        user = int(ff[0])
        item = int(ff[1])
        like = classes[ff[3]]
        user_history[user].append((item,like))

    count = 0
    for user,history in user_history.iteritems():
        items = []
        likes = []
        next_items = []
        next_likes = []
        w = []
        for i in xrange(-1,len(history)-1):
            if i > -1:
                item, like = history[i]
                y = [0, 0, 0, 0]
                y[like] = 1
                items.append(item)
                likes.append(y)
            else:
                items.append(len(item_map))
                likes.append([0, 0, 0, 0])

            next_item, next_like = history[i+1]
            y = [0, 0, 0, 0]
            y[next_like] = 1
            next_items.append(next_item)
            next_likes.append(y)
            w.append(1.)

        #if len(history) - 1 < SEQ_LENGTH:

        left_train.append(items)
        left_x_train.append(likes)
        right_train.append(next_items)
        y_train.append(next_likes)
        w_train.append(w)

    left_train = sequence.pad_sequences(np.array(left_train), maxlen=SEQ_LENGTH)
    left_x_train = sequence.pad_sequences(np.array(left_x_train), maxlen=SEQ_LENGTH)
    right_train = sequence.pad_sequences(np.array(right_train), maxlen=SEQ_LENGTH)
    y_train = sequence.pad_sequences(np.array(y_train), maxlen=SEQ_LENGTH)
    w_train = sequence.pad_sequences(np.array(w_train), maxlen=SEQ_LENGTH)
    return left_train, left_x_train, right_train, y_train, w_train


def load_test_set(st_path, tg_path):
    user_history = defaultdict(list)
    user_targets = defaultdict(list)

    left_valid = []
    left_x_valid = []
    right_valid = []

    for line in open(st_path):
        ff = line.strip().split(',')
        user = int(ff[0])
        item = int(ff[1])
        like = classes[ff[3]]
        user_history[user].append((item,like))

    for line in open(tg_path):
        ff = line.strip().split(',')
        user = int(ff[0])
        item = int(ff[1])
        user_targets[user].append(item)

    count = 0
    for user,targets in user_targets.iteritems():
        history = user_history[user]
        items = []
        likes = []
        next_items = []
        for i in xrange(-1,len(history)):
            if i >= 0:
                item, like = history[i]
                y = [0, 0, 0, 0]
                y[like] = 1
                items.append(item)
                likes.append(y)
            else:
                items.append(len(item_map))
                likes.append([0,0,0,0])
            if i < len(history)-1:
                next_items.append(0)

        for i in xrange(0,len(targets)):
            next_items_2 = list(next_items)
            next_items_2.append(targets[i])
            left_valid.append(items)
            right_valid.append(next_items_2)
            left_x_valid.append(likes)

    left_valid = sequence.pad_sequences(np.array(left_valid), maxlen=SEQ_LENGTH)
    left_x_valid = sequence.pad_sequences(np.array(left_x_valid), maxlen=SEQ_LENGTH)
    right_valid = sequence.pad_sequences(np.array(right_valid), maxlen=SEQ_LENGTH)
    return left_valid, left_x_valid, right_valid

def load_valid_set(st_path, tg_path):
    user_history = defaultdict(list)
    user_targets = defaultdict(list)

    left_valid = []
    left_x_valid = []
    right_valid = []
    y_valid = []
    w_valid = []

    for line in open(st_path):
        ff = line.strip().split(',')
        user = int(ff[0])
        item = int(ff[1])
        like = classes[ff[3]]
        user_history[user].append((item,like))

    for line in open(tg_path):
        ff = line.strip().split(',')
        user = int(ff[0])
        item = int(ff[1])
        like = classes[ff[3]]
        user_targets[user].append((item,like))

    count = 0
    for user,targets in user_targets.iteritems():
        history = user_history[user]
        items = []
        likes = []
        ans = []
        next_items = []
        next_likes = []
        w = []
        for i in xrange(-1,len(history)):
            if i >= 0:
                item, like = history[i]
                y = [0, 0, 0, 0]
                y[like] = 1
                items.append(item)
                likes.append(y)
            else:
                items.append(len(item_map))
                likes.append([0,0,0,0])
            if i < len(history)-1:
                next_items.append(0)
                ans.append([0,0,0,0])
                w.append(0.)
        w.append(1.)

        for i in xrange(0,len(targets)):
            next_items_2 = list(next_items)
            next_items_2.append(targets[i][0])
            left_valid.append(items)
            right_valid.append(next_items_2)
            left_x_valid.append(likes)
            y = [0, 0, 0, 0]
            y[targets[i][1]] = 1
            ans_2 = list(ans)
            ans_2.append(y)
            y_valid.append(ans_2)
            w_valid.append(w)

    left_valid = sequence.pad_sequences(np.array(left_valid), maxlen=SEQ_LENGTH)
    left_x_valid = sequence.pad_sequences(np.array(left_x_valid), maxlen=SEQ_LENGTH)
    right_valid = sequence.pad_sequences(np.array(right_valid), maxlen=SEQ_LENGTH)
    y_valid = sequence.pad_sequences(np.array(y_valid), maxlen=SEQ_LENGTH)
    w_valid = sequence.pad_sequences(np.array(w_valid), maxlen=SEQ_LENGTH)
    return left_valid, left_x_valid, right_valid, y_valid, w_valid



if __name__ == '__main__':
    EMBED_SIZE=8
    LSTM_SIZE=64
    RELU_SIZE=32
    LSTM_DROPOUT=0.2
    BATCH_SIZE=256
    EPOCHS=40
    #MAX_USER_ID=1103495
    #MAX_ITEM_ID=1100028
    MAX_USER_ID=35000
    MAX_ITEM_ID=1000
    SEQ_LENGTH=30

    classes = {'dislike' : 0, 'skip' : 1, 'view' : 2, 'like' : 3}

    np.random.seed(2)

    user_map, item_map = load_users_and_items(sys.argv[4], sys.argv[5])
    #print len(user_map), len(item_map)
    print "Load train"
    left_train, left_x_train, right_train, y_train, w_train = load_set(sys.argv[1])
    print "Load test"
    left_test, left_x_test, right_test, y_test, w_test = load_valid_set(sys.argv[2], sys.argv[3])

    if True:
        print "params:",EMBED_SIZE,LSTM_SIZE,RELU_SIZE,LSTM_DROPOUT,BATCH_SIZE,EPOCHS
        left_input = Input(shape=(SEQ_LENGTH,), dtype='int32')
        left_embed = Embedding(len(item_map) + 1, EMBED_SIZE, input_shape=(SEQ_LENGTH,))(left_input)
        left_x = Input(shape=(SEQ_LENGTH,4), dtype='float32')
        left_concat = Concatenate(axis=2)([left_embed, left_x])

        lstm = LSTM(LSTM_SIZE, dropout=LSTM_DROPOUT, recurrent_dropout=LSTM_DROPOUT, return_sequences=True)(left_concat)
        #lstm = CuDNNGRU(EMBED_SIZE, return_sequences=True)(left_concat)
        #lstm_dense = Dense(EMBED_SIZE)(lstm)

        right_input = Input(shape=(SEQ_LENGTH,), dtype='int32')
        #right_embed = Dropout(1.0)(Embedding(len(item_map) + 1, EMBED_SIZE, input_shape=(SEQ_LENGTH,))(right_input))
        right_embed = Dropout(1.0)(Embedding(len(item_map) + 1, LSTM_SIZE, input_shape=(SEQ_LENGTH,))(right_input))
        #right_embed = Embedding(len(item_map) + 1, LSTM_SIZE, input_shape=(SEQ_LENGTH,))(right_input)

        dot_layer = Lambda(lambda embeds: K.sum(tf.multiply(embeds[0], embeds[1][:tf.newaxis]), axis=-1, keepdims=True))([lstm, right_embed])
        #dot_layer = Dot(axes=2)([lstm, right_embed])
        cat_layer = Concatenate(axis=2)([lstm, right_embed, dot_layer])

        relu_layer =  Dense(RELU_SIZE, activation='relu')(cat_layer)
        dense_layer = Dense(4, activation='softmax')(relu_layer)

        model = Model(inputs=[left_input, left_x, right_input], outputs=dense_layer)
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      sample_weight_mode="temporal",
                      metrics=['accuracy'])
        #print model.summary()



        print "Model fit"
        model.fit([left_train, left_x_train, right_train],
            y_train,
            batch_size=BATCH_SIZE,
            nb_epoch=EPOCHS,
            #validation_data=([left_test, left_x_test, right_test], y_test),
            validation_split=0.2,
            sample_weight=w_train,
            verbose=1)

        pred = model.predict([left_test, left_x_test, right_test])
        s = 0.
        n = 0.
        for i in xrange(0,len(left_test)):
            #print i
            #print left_test[i]
            #print left_x_test[i]
            #print right_test[i][-1]
            #print y_test[i][-1]
            #print pred[i][-1]
            r = y_test[i][-1]
            p = pred[i][-1]

            e = -10*p[0] - 0.1*p[1] + 0.1*p[2] + 0.5*p[3]
            v = -10*r[0] - 0.1*r[1] + 0.1*r[2] + 0.5*r[3]
            if e > 0:
                s += v
            else:
                s -= v
            n += 1.
            #if n % 100 == 0:
            #    print r, p, e, v
            #    print s/n, n
        print "SCORE",s/n, n


    #model.save(sys.argv[6])
