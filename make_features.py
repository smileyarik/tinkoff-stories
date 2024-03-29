import pickle
import sys
import json
from profiles import *
from make_profiles import *
from learn_lstm import *
import numpy as np
import tensorflow as tf
#tf.enable_eager_execution()
import time
import os
from keras.models import load_model
from learn_nn2 import tf_score, load_users_and_items

from keras.utils.generic_utils import get_custom_objects
get_custom_objects().update({"tf_score": tf_score})

def date_to_timestamp(strdate):
    # windows doesn't support "%s", so let's use explicit calculation for timestamp
    epoch = datetime.datetime(1970,1,1)
    event_time = datetime.datetime.strptime(strdate, "%Y-%m-%d %H:%M:%S")
    return int((event_time - epoch).total_seconds())

print("Loading")
users = pickle.load(open(sys.argv[1], 'rb'))
items = pickle.load(open(sys.argv[2], 'rb'))

known_target = (sys.argv[3] == 'train')
start_ts = date_to_timestamp(sys.argv[4])

feat_out = open(sys.argv[6], 'w')

user_map, item_map = load_users_and_items(sys.argv[8], sys.argv[9])
model = load_model(sys.argv[7])
model2 = load_model(sys.argv[11])
#model_lstm = load_model(sys.argv[12])
#left_lstm, left_x_lstm, right_lstm = load_test_set(sys.argv[13], sys.argv[5])

descr = {}
with open(sys.argv[10]) as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader, None)
    for row in reader:
        item_id = int(row[0])
        descr[item_id] = row[1]
        #pages = row[1].count('story-page')

weights = {'like' : 0.5, 'view' : 0.1, 'skip' : -0.1, 'dislike' : -10}
weights2 = {CT_LIKE : 0.5, CT_VIEW : 0.1, CT_SKIP : -0.1, CT_DISLIKE : -10}

def make_counters():
    return Counters()

def try_div(a, b, default):
    if b > 0:
        return float(a)/b
    else:
        return default

def cos_prob(user, item, ot_type, user_ct_type, event_ct_type, show_ct_type, rt_type, ts):
    user_mod = 0.
    item_mod = 0.
    prod = 0.

    user_slice = user.slice(ot_type, user_ct_type, rt_type)

    event_slice = item.slice(ot_type, event_ct_type, rt_type)
    show_slice = item.slice(ot_type, show_ct_type, rt_type)

    for key,c in user_slice.items():
        v = c.get(ts, rt_type)

        user_mod += v

    for key,c in show_slice.items():
        v = c.get(ts, rt_type)

        item_mod += v*v
        if key in user_slice and key in event_slice:
            prod += (user_slice[key].get(ts, rt_type) * event_slice[key].get(ts, rt_type) / float(v) if v > 0 else 0.)

    if user_mod == 0 or item_mod == 0:
        return -100.

    if prod > 0:
        prod = prod / user_mod

    return prod


def counter_cos(user, item, ot_type, user_ct_type, item_ct_type, rt_type, ts, dbg=False):
    user_mod = 0.
    item_mod = 0.
    prod = 0.

    user_slice = user.slice(ot_type, user_ct_type, rt_type)
    item_slice = item.slice(ot_type, item_ct_type, rt_type)

    if dbg:
        print("=========")
        print("User:)")
    for key,c in user_slice.items():
        v = c.get(ts, rt_type)
        if dbg:
            print(key, v)

        user_mod += v*v

    if dbg:
        print("Item:")
    for key,c in item_slice.items():
        v = c.get(ts, rt_type)
        if dbg:
            print(key, v)

        item_mod += v*v
        if key in user_slice:
            v2 = user_slice[key].get(ts, rt_type)
            prod += v * v2

    if user_mod == 0 or item_mod == 0:
        return -100.

    if prod > 0:
        prod = prod / (math.sqrt(user_mod) * math.sqrt(item_mod))

    if dbg:
        print("Cos:", prod)

    return prod


print("Calc global stat")
full_events = Counters()
for item_id, item in items.items():
    for event in [CT_LIKE, CT_VIEW, CT_SKIP, CT_DISLIKE, CT_SHOW]:
        for rt in [RT_SUM, RT_7D, RT_30D]:
            full_events.update_from(item, OT_GLOBAL, event, rt, event, rt, start_ts)

def iter_rows(path):
    idx = 0
    with open(path) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        if not known_target:
            next(reader, None)
        for row in reader:
            if (len(row)) != 0: # stupid windows
                yield (idx, row)
                idx += 1


print("Apply nnet")

user_ids = []
user_genders = []
user_ages = []
user_marital_statuses = []
user_jobs = []
item_ids = []

for (idx, row) in iter_rows(sys.argv[5]):
    user_id = int(row[0])
    item_id = int(row[1])

    u = user_map[user_id]
    user_ids.append(u.idx)
    user_genders.append(u.gender)
    user_ages.append(u.age)
    user_marital_statuses.append(u.marital_status)
    user_jobs.append(u.job)
    item_ids.append(item_map[item_id])

x = [np.array(l) for l in (user_ids, user_genders, user_ages, user_marital_statuses, user_jobs, item_ids)]
nnet_predictions = model.predict([np.array(user_ids), np.array(item_ids)])
nnet2_predictions = model2.predict(x)

print("Calc features")
for (idx, row) in iter_rows(sys.argv[5]):
    user_id = int(row[0])
    item_id = int(row[1])
    user = users[int(row[0])]
    item = items[int(row[1])]
    ts = date_to_timestamp(row[2])
    target = 0
    pid = 0
    if known_target:
        target = weights[row[3]]
    else:
        pid = int(row[3])

    f = []

    user_size = float(user.get(OT_GLOBAL, CT_SHOW, RT_SUM, '', ts))
    f.append(user_size) #0
    s = 0
    for event in [CT_LIKE, CT_VIEW, CT_SKIP, CT_DISLIKE]:
    #    for rt in [RT_SUM, RT_7D, RT_30D]:
        for rt in [RT_SUM]:
            f.append(try_div(user.get(OT_GLOBAL, event, rt, '', ts),user.get(OT_GLOBAL, CT_SHOW, rt, '', ts),-100.)) # 1-4
            s += user.get(OT_GLOBAL, event, rt, '', ts) * weights2[event]
    f.append(try_div(s, user_size, -100)) # 5

    item_size = float(item.get(OT_GLOBAL, CT_SHOW, RT_SUM, '', ts))
    f.append(item_size) # 6
    s = 0
    for event in [CT_LIKE, CT_VIEW, CT_SKIP, CT_DISLIKE]:
        #for rt in [RT_SUM, RT_7D, RT_30D]:
        for rt in [RT_SUM]:
            f.append(try_div(item.get(OT_GLOBAL, event, rt, '', ts),item.get(OT_GLOBAL, CT_SHOW, rt, '', ts),-100.)) # 7-10
            s += item.get(OT_GLOBAL, event, rt, '', ts) * weights2[event]
    f.append(try_div(s, item_size, -100)) # 11

    for ot in [OT_GENDER, OT_AGE, OT_JOB, OT_MARITAL, OT_PRODUCT, OT_MCC]: # 12-41
        user_ct = CT_HAS if ot != OT_MCC else CT_TRANSACTION

        #def cos_prob(user, item, ot_type, user_ct_type, event_ct_type, show_ct_type, rt_type, ts):
        p_like = cos_prob(user, item, ot, user_ct, CT_LIKE, CT_SHOW, RT_SUM, ts)
        p_view = cos_prob(user, item, ot, user_ct, CT_VIEW, CT_SHOW, RT_SUM, ts)
        p_skip = cos_prob(user, item, ot, user_ct, CT_SKIP, CT_SHOW, RT_SUM, ts)
        p_dislike = cos_prob(user, item, ot, user_ct, CT_DISLIKE, CT_SHOW, RT_SUM, ts)
        w = 0.5 * p_like + 0.1 * p_view - 0.1 * p_skip - 10 * p_dislike
        f.append(p_like)
        f.append(p_view)
        f.append(p_skip)
        f.append(p_dislike)
        f.append(w)

    # 42-51
    for pred in (nnet_predictions, nnet2_predictions):
        y = pred[idx]
        for i in range(4):
            f.append(y[i])
        f.append(-10*y[0] - 0.1*y[1] + 0.1*y[2] + 0.5*y[3])

    # 52, 53
    for t in ['M', 'F']:
        f.append(user.get(OT_GENDER, CT_HAS, RT_SUM, t, ts))

    age = 0
    for t in [0,15,20,25,30,35,40,45,50,55,60,65,70]:
        if user.get(OT_AGE, CT_HAS, RT_SUM, t, ts) != 0:
            age = t

    # 54
    f.append(age)

    if False:
        for t in ['M', 'F']:
            f.append(user.get(OT_GENDER, CT_HAS, RT_SUM, t, ts))

        for t in [0,15,20,25,30,35,40,45,50,55,60,65,70]:
            f.append(user.get(OT_AGE, CT_HAS, RT_SUM, t, ts))

        for t in ['', 'CIV', 'DIV', 'MAR', 'UNM', 'WID']:
            f.append(user.get(OT_MARITAL, CT_HAS, RT_SUM, t, ts))

        for t in range(0,23):
            f.append(user.get(OT_JOB, CT_HAS, RT_SUM, str(t), ts))

        f.append(ts - start_ts) #90
        f.append(descr[item_id].count('story-page') if item_id in descr else 0.) # 91
        f.append(descr[item_id].count('tinkoffbank:') if item_id in descr else 0.) # 92

        for ct in [CT_LIKE, CT_VIEW, CT_SKIP, CT_DISLIKE]: # 12-35
            #def counter_cos(user, item, ot_type, user_ct_type, item_ct_type, rt_type, ts):
            #f.append(counter_cos(user, item, OT_MCC, CT_TRANSACTION, ct, RT_SUM, ts)) #
            #f.append(counter_cos(user, item, OT_GENDER, CT_HAS, ct, RT_SUM, ts, True))
            #user_slice = user.slice(OT_AGE, CT_HAS, RT_SUM)
            #item_slice = item.slice(OT_AGE, ct, RT_SUM)
            #f.append(counter_cos(user, item, OT_AGE, CT_HAS, ct, RT_SUM, ts))
            #f.append(len(user_slice))
            #f.append(len(item_slice))
            #f.append(counter_cos(user, item, OT_JOB, CT_HAS, ct, RT_SUM, ts))
            #f.append(counter_cos(user, item, OT_MARITAL, CT_HAS, ct, RT_SUM, ts))
            #f.append(counter_cos(user, item, OT_PRODUCT, CT_HAS, ct, RT_SUM, ts))
            pass

    feat_out.write('%d\t%d\t%f\t%d\t%s\n' % (user_id, item_id, target, pid, '\t'.join([str(ff) for ff in f])))

exit()



