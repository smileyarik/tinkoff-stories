import pickle
import sys
import json
from profiles import *
from make_profiles import *
import numpy as np
#import tensorflow as tf
#tf.enable_eager_execution()
import time
import os
#from learn_lstm import make_weeked_visits

print "Loading"
users = pickle.load(open(sys.argv[1]))
items = pickle.load(open(sys.argv[2]))

known_target = (sys.argv[3] == 'train')
start_ts = int(datetime.datetime.strptime(sys.argv[4], "%Y-%m-%d %H:%M:%S").strftime("%s"))

feat_out = open(sys.argv[6], 'w')

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

    for key,c in user_slice.iteritems():
        v = c.get(ts, rt_type)

        user_mod += v

    for key,c in show_slice.iteritems():
        v = c.get(ts, rt_type)

        item_mod += v*v
        if key in user_slice and key in event_slice:
            prod += user_slice[key].get(ts, rt_type) * event_slice[key].get(ts, rt_type) / float(v)

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
        print "========="
        print "User:"
    for key,c in user_slice.iteritems():
        v = c.get(ts, rt_type)
        if dbg:
            print key, v

        user_mod += v*v

    if dbg:
        print "Item:"
    for key,c in item_slice.iteritems():
        v = c.get(ts, rt_type)
        if dbg:
            print key, v

        item_mod += v*v
        if key in user_slice:
            v2 = user_slice[key].get(ts, rt_type)
            prod += v * v2

    if user_mod == 0 or item_mod == 0:
        return -100.

    if prod > 0:
        prod = prod / (math.sqrt(user_mod) * math.sqrt(item_mod))

    if dbg:
        print "Cos:", prod

    return prod


print "Calc global stat"
full_events = Counters()
for item_id, item in items.iteritems():
    for event in [CT_LIKE, CT_VIEW, CT_SKIP, CT_DISLIKE, CT_SHOW]:
        for rt in [RT_SUM, RT_7D, RT_30D]:
            full_events.update_from(item, OT_GLOBAL, event, rt, event, rt, start_ts)

print "Calc features"
with open(sys.argv[5]) as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    if not known_target:
        next(reader, None)
    for row in reader:
        user_id = int(row[0])
        item_id = int(row[1])
        user = users[int(row[0])]
        item = items[int(row[1])]
        ts = int(datetime.datetime.strptime(row[2], "%Y-%m-%d %H:%M:%S").strftime("%s"))
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

        for ot in [OT_GENDER, OT_AGE, OT_JOB, OT_MARITAL, OT_PRODUCT, OT_MCC]:
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

        for t in ['M', 'F']:
            f.append(user.get(OT_GENDER, CT_HAS, RT_SUM, t, ts))

        for t in [0,15,20,25,30,35,40,45,50,55,60,65,70]:
            f.append(user.get(OT_AGE, CT_HAS, RT_SUM, t, ts))

        for t in ['', 'CIV', 'DIV', 'MAR', 'UNM', 'WID']:
            f.append(user.get(OT_MARITAL, CT_HAS, RT_SUM, t, ts))

        for t in xrange(0,23):
            f.append(user.get(OT_JOB, CT_HAS, RT_SUM, str(t), ts))

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



user_als = pickle.load(open(sys.argv[3]))
item_als = pickle.load(open(sys.argv[4]))

print "Find target users"
target_users = defaultdict(lambda:set())
data = json.load(open(sys.argv[6]))
for user in data['users']:
    target_users[int(user)] = set()

if sys.argv[5] != "-":
    with open(sys.argv[5]) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')

        for row in reader:
            if int(row[1]) in target_users:
                target_users[int(row[1])].add(int(row[0]))

ts = float(sys.argv[8]) * 6

print "Find top movies"
top = []
for item_id,counters in items.iteritems():
    #print item_id, counters.get(OT_GLOBAL, CT_TRANSACTION, RT_SUM, '', ts)
    top.append((item_id, counters.get(OT_GLOBAL, CT_TRANSACTION, RT_30D, '', ts)))
top = sorted(top, key=lambda x:x[1], reverse=True)
print top[0:200]

feat_out = open(sys.argv[7], 'w')

def counter_cos(user, item, ot_type, user_ct_type, item_ct_type, rt_type, ts):
    user_mod = 0.
    item_mod = 0.
    prod = 0.

    user_slice = user.slice(ot_type, user_ct_type, rt_type)
    item_slice = item.slice(ot_type, item_ct_type, rt_type)

    for key,c in user_slice.iteritems():
        v = c.get(ts, rt_type)
        user_mod += v*v

    for key,c in item_slice.iteritems():
        v = c.get(ts, rt_type)
        item_mod += v*v
        if key in user_slice:
            v2 = user_slice[key].get(ts, rt_type)
            prod += v * v2

    if prod > 0:
        prod = prod / (math.sqrt(user_mod) * math.sqrt(item_mod))

    return prod

vocab_size = 10229
max_id = 10199
embedding_dim = 256
rnn_units = 1024

rnn = tf.keras.layers.CuDNNGRU

checkpoint_dir = sys.argv[9]

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
        rnn(rnn_units,
            return_sequences=True,
            recurrent_initializer='glorot_uniform',
            stateful=True),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model

model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1, None]))

print "Calc global stat"
full_visits = defaultdict(float)
full_rated = 0.
for item_id, item in items.iteritems():
    full_rated += item.get(OT_GLOBAL, CT_HAS_RATING, RT_SUM, '', ts)
    for rt in [RT_SUM, RT_7D, RT_30D]:
        full_visits[rt] += item.get(OT_GLOBAL, CT_TRANSACTION, rt, '', ts)

print "Make features"
for user_id, positives in target_users.iteritems():
    user = users[user_id]
    filtered = user.slice(OT_ITEM, CT_TRANSACTION, RT_SUM)

    user_visits = []
    for item_id,counter in filtered.iteritems():
        user_visits.append((item_id, counter.ts))

    prob = []
    if len(user_visits) > 0:
        user_visits = sorted(user_visits, key=lambda x:x[1])
        user_visits = make_weeked_visits(user_visits, ts, 0)
        if user_id % 5000 == 0:
            print user_id
            print user_visits

        input_eval = tf.expand_dims(user_visits, 0)
        model.reset_states()
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        prob = predictions.numpy()[-1]

        toptop = sorted([(i,prob[i]) for i in xrange(1,max_id+1)], key=lambda x:x[1], reverse=True)[0:200]
    else:
        toptop = top[0:200]

    top_set = set([x[0] for x in toptop])
    for item_id,counters in items.iteritems():
        if users[user_id].has(OT_ITEM, CT_BOOKMARK, RT_SUM, item_id) and item_id not in top_set:
            toptop.append((item_id, prob[item_id] if len(user_visits) > 0 else 0.))

    for item_id,baserank in toptop:
        if item_id in filtered:
            continue

        item = items[item_id]
        f = []

        user_size = float(user.get(OT_GLOBAL, CT_TRANSACTION, RT_SUM, '', ts))

        f.append(user_size) #0
        f.append(item.get(OT_GLOBAL, CT_TRANSACTION, RT_SUM, '', ts)/full_visits[RT_SUM])
        f.append(item.get(OT_GLOBAL, CT_TRANSACTION, RT_7D, '', ts)/full_visits[RT_7D])
        f.append(item.get(OT_GLOBAL, CT_TRANSACTION, RT_30D, '', ts)/full_visits[RT_30D])
        f.append(item.get(OT_GLOBAL, CT_HAS_RATING, RT_SUM, '', ts)/full_rated)
        f.append(item.get(OT_GLOBAL, CT_RATING, RT_SUM, '', ts)/f[-1] if f[-1] > 0 else 0.) #5

        for t in ['movie', 'series']:
            f.append(item.get(OT_ITEMTYPE, CT_HAS, RT_SUM, t, ts))

        for t in ['subscription', 'purchase', 'rent']:
            f.append(item.get(OT_AVAIL, CT_HAS, RT_SUM, t, ts)) # 8 - 10

        f.append(ts - item.get(OT_FEATURE, CT_VALUE, RT_SUM, 1, ts) * 6)
        for i in xrange(2,6):
            f.append(item.get(OT_FEATURE, CT_VALUE, RT_SUM, i, ts)) # 12 - 15

        f.append(item.get(OT_DURATION, CT_VALUE, RT_SUM, '', ts))

        f.append(user.get(OT_DURATION, CT_TRANSACTION, RT_SUM, '', ts)/user_size if user_size > 0 else -100.)
        f.append(user.get(OT_WATCHTIME, CT_TRANSACTION, RT_SUM, '', ts)/user_size if user_size > 0 else -100.)

        for ct in [CT_TRANSACTION, CT_BOOKMARK, CT_HAS_RATING]:
            f.append(counter_cos(user, item, OT_ATTR, ct, CT_HAS, RT_SUM, ts)) #19, 22, 25
            f.append(counter_cos(user, item, OT_AVAIL, ct, CT_HAS, RT_SUM, ts)) #20, 23, 26
            f.append(counter_cos(user, item, OT_ITEMTYPE, ct, CT_HAS, RT_SUM, ts))

        f.append(counter_cos(user, item, OT_DEVICETYPE, CT_TRANSACTION, CT_TRANSACTION, RT_SUM, ts)) # 28
        f.append(counter_cos(user, item, OT_DEVICEMAN, CT_TRANSACTION, CT_TRANSACTION, RT_SUM, ts))
        f.append(counter_cos(user, item, OT_WATCHMODE, CT_TRANSACTION, CT_TRANSACTION, RT_SUM, ts)) # 30

        als = 0.
        if user_id in user_als and item_id in item_als:
            als = np.dot(user_als[user_id], item_als[item_id])

        f.append(als) # 31

        f.append(1 if user.has(OT_ITEM, CT_BOOKMARK, RT_SUM, item_id) else 0)
        f.append(ts - user.getts(OT_ITEM, CT_BOOKMARK, RT_SUM, item_id) if user.has(OT_ITEM, CT_BOOKMARK, RT_SUM, item_id) else 0)
        f.append(1 if user.has(OT_ITEM, CT_RATING, RT_SUM, item_id) else 0)
        f.append(user.get(OT_ITEM, CT_RATING, RT_SUM, item_id, ts) if user.has(OT_ITEM, CT_RATING, RT_SUM, item_id) else 0.) # 35

        f.append(float(item.get(OT_GLOBAL, CT_TRANSACTION, RT_7D, '', ts))/item.get(OT_GLOBAL, CT_TRANSACTION, RT_30D, '', ts) if item.get(OT_GLOBAL, CT_TRANSACTION, RT_30D, '', ts) > 0. else 0.) # 36

        if len(user_visits) > 0:
            f.append(baserank)
        else:
            f.append(0.)

        target = 0
        if item_id in positives:
            target = 1
        feat_out.write('%d\t%d\t%d\t%s\n' % (user_id, item_id, target, '\t'.join([str(ff) for ff in f])))


