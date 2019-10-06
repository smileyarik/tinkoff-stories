import os
import sys
import csv

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import keras.backend as K
from keras.layers import Input, Embedding, LSTM, Dense, Dot, Reshape, Flatten, Concatenate, Dropout, Lambda, Layer, Add, Activation
from keras.models import Model
import keras
import numpy as np

import tensorflow as tf
from sklearn.metrics import roc_auc_score
from collections import defaultdict

genders = ['M', 'F']
num_ages = 15
marital_statuses = ['CIV', 'DIV', 'MAR', 'UNM', 'WID']
num_jobs = 24

def lookup(what, where):
    for i in range(len(where)):
        if where[i] == what:
            return i + 1
    return 0

class User:
    __slots__ = ['idx', 'gender', 'age', 'marital_status', 'job']
    def __init__(self, idx, gender, age, marital_status, job):
        self.idx = idx
        self.gender = gender
        self.age = age
        self.marital_status = marital_status
        self.job = job

def load_users_and_items(user_file, item_file):
    user_map = {}
    i = 1
    with open(user_file, encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader, None)
        for row in reader:
            if (len(row)) == 0: continue # stupid windows
            user_id = int(row[0])
            user_map[user_id] = User(
                i,
                lookup(row[8], genders),
                min(num_ages, (1 + int(float(row[9]) / 5)) if row[9] != '' else 0),
                lookup(row[10], marital_statuses),
                min(num_jobs, (1 + int(row[13])) if row[13] != '' else 0),
            )
            i += 1

    item_map = {}
    i = 1
    with open(item_file, encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            if (len(row)) == 0: continue # stupid windows
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
    max_sum = 0.
    for i in range(0,len(y_true)):
        s = 0.
        r = 0.
        for j in range(0,4):
            s += y_pred[i][j] * weight[j]
            r += y_true[i][j] * weight[j]
        if s > 0:
            res += r
        else:
            res -= r
    return float(res) / len(y_true)

def tf_score(y_true, y_pred):
    return tf.py_func(score, (y_true, y_pred), tf.double)

if __name__ == '__main__':
    EMBED_SIZE=8

    classes = {'dislike' : 0, 'skip' : 1, 'view' : 2, 'like' : 3}

    np.random.seed(2)

    user_map, item_map = load_users_and_items(sys.argv[3], sys.argv[4])
    print(len(user_map), len(item_map))

    embedding_droupout_rate = 0.2

    user_input = Input(shape=(1,), dtype='int32', name="user_idx")
    user_embed = Embedding(len(user_map) + 1, EMBED_SIZE, input_length=1, embeddings_initializer='zeros')(user_input)
    user_embed = Dropout(embedding_droupout_rate)(user_embed)

    gender_input = Input(shape=(1,), dtype='int32', name="gender")
    gender_embed = Dropout(embedding_droupout_rate)(Embedding(len(genders) + 1, EMBED_SIZE, input_length = 1)(gender_input))

    age_input = Input(shape=(1,), dtype='int32', name="age")
    age_embed = Dropout(embedding_droupout_rate)(Embedding(num_ages + 1, EMBED_SIZE, input_length = 1)(age_input))

    marital_status_input = Input(shape=(1,), dtype='int32', name="mar_status")
    marital_status_embed = Dropout(embedding_droupout_rate)(Embedding(len(marital_statuses) + 1, EMBED_SIZE, input_length = 1)(marital_status_input))

    job_input = Input(shape=(1,), dtype='int32', name="job")
    job_embed = Dropout(embedding_droupout_rate)(Embedding(num_jobs + 1, EMBED_SIZE, input_length = 1)(job_input))

    initial_user_embed_layers = [user_embed, gender_embed, age_embed, marital_status_embed, job_embed]
    full_user_embed = Add()(initial_user_embed_layers)

    item_input = Input(shape=(1,), dtype='int32')
    item_embed = Dropout(embedding_droupout_rate)(Embedding(len(item_map) + 1, EMBED_SIZE, input_length=1)(item_input))
    
    dot_layers = []
    for i in range(4):
        user_dense = Dropout(0.2)(Flatten()(Dense(EMBED_SIZE)(full_user_embed)))
        item_dense = Dropout(0.2)(Flatten()(Dense(EMBED_SIZE)(item_embed)))
        dot_layer = Dot(axes=1)([user_dense, item_dense])
        dot_layers.append(dot_layer)

    cat_layer = Concatenate(axis=1)(dot_layers)
    output_layer = Activation('softmax')(cat_layer)

    model = Model(
        inputs=[user_input, gender_input, age_input, marital_status_input, job_input, item_input],
        outputs=output_layer)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy', tf_score])

    def load(path):
        users = []
        genders = []
        ages = []
        marital_statuses = []
        jobs = []
        items = []
        ys = []

        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if len(line) == 0: continue # stupid windows
            ff = line.split(',')
            user_id = int(ff[0])
            item = int(ff[1])
            user = user_map[user_id]
            users.append(user.idx)
            genders.append(user.gender)
            ages.append(user.age)
            marital_statuses.append(user.marital_status)
            jobs.append(user.job)
            items.append([item_map[item]])
            y = [0, 0, 0, 0]
            y[classes[ff[3]]] = 1
            ys.append(y)

        return users, genders, ages, marital_statuses, jobs, items, ys

    user_train, gender_train, age_train, marital_statuses_train, jobs_train, item_train, y_train = load(sys.argv[1])
    user_test, gender_test, age_test, marital_statuses_test, jobs_test, item_test, y_test = load(sys.argv[2])

    print(len(y_train), sum(sum(x) for x in y_train))
    print(len(y_test), sum(sum(x) for x in y_test))
    print(np.array(user_train).shape)
    print(np.array(y_train).shape)

    model.fit(
        [np.array(x) for x in (user_train, gender_train, age_train, marital_statuses_train, jobs_train, item_train)],
        np.array(y_train),
        batch_size=1024,
        nb_epoch=10,
        validation_data=(
            [np.array(x) for x in (user_test, gender_test, age_test, marital_statuses_test, jobs_test, item_test)],
            np.array(y_test)),
        #validation_split=0.2,
        verbose=2)#

    with open('predict.nn', 'w') as p_out:
        predicts = model.predict([np.array(x) for x in (user_test, gender_test, age_test, marital_statuses_test, jobs_test, item_test)])
        sum_target = 0.
        sum_score = 0.
        for i in range(len(y_test)):
            t = y_test[i]
            y = predicts[i]
            target = -10.*t[0] - 0.1*t[1] + 0.1*t[2] + 0.5*t[3]
            predict = -10.*y[0] - 0.1*y[1] + 0.1*y[2] + 0.5*y[3]
            sum_target += target if target > 0 else -target
            sum_score += target * (1 if predict > 0 else -1)
            p_out.write("%f\t%f\t%s\n" % (target, predict, "\t".join([str(yy) for yy in y])))
        print(sum_target, sum_score, sum_target/len(y_test), sum_score/len(y_test), sum_score / sum_target)
        print(score(y_test, predicts))
        print(score(predicts, y_test))
        



    model.save(sys.argv[5])
