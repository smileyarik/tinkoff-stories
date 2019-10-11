import json
import time
import random
import sys
import math
from six.moves import xrange
from catboost import Pool, CatBoostClassifier

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

class CustomMetric(object):
    def get_final_error(self, error, weight):
        return error / (weight + 1e-38)

    def is_max_optimal(self):
        return True

    def evaluate(self, approxes, target, weight):
        #print len(approxes), len(approxes[0]), len(target)
        # approxes - list of list-like objects (one object per approx dimension)
        # target - list-like object
        # weight - list-like object, can be None
        #scores = {0 : -10., 1 : -0.1, 2 : 0.1, 3 : 0.5}
        scores = {0 : -10., 1 : -0.1, 2 : 0.1, 3 : 0.5}
        error_sum = 0.
        weight_sum = 0.
        for i in xrange(0,len(target)):
            e = 0
            exp_sum = 0
            for j in xrange(0,4):
                exp_sum += math.exp(approxes[j][i])
            for j in xrange(0,4):
                e += (math.exp(approxes[j][i]) / exp_sum) * scores[j]
            #if i % 10000 == 0:
            #    print [approxes[x][i] for x in xrange(0,4)]
            #    print [math.exp(approxes[x][i]) / exp_sum for x in xrange(0,4)], target[i], e

            if e > 0:
                error_sum += float(scores[int(target[i])])
            else:
                error_sum -= float(scores[int(target[i])])
            weight_sum += 1

        #print error_sum, weight_sum
        return error_sum, weight_sum

train_pool = Pool(data=sys.argv[1], column_description='cd')
eval_pool = Pool(data=sys.argv[2], column_description='cd')

# Initialize CatBoostClassifier with custom `loss_function`
#model = CatBoostClassifier(loss_function=LoglossObjective(),
#                           eval_metric=LoglossObjective(),
#                           iterations=int(sys.argv[4]),
#                           ignored_features=[])
model = CatBoostClassifier(loss_function='MultiClass',
                           #eval_metric=CustomMetric(),
                           iterations=int(sys.argv[4]),
                           ignored_features=[])
                           #ignored_features=xrange(46,93))

# Fit model
model.fit(train_pool, eval_set=eval_pool)
# Only prediction_type='RawFormulVal' allowed with custom `loss_function`
preds_raw = model.predict(eval_pool,
                          prediction_type='Probability')
preds_class = model.predict(eval_pool,
                          prediction_type='Class')

model.save_model(sys.argv[3])

print model.get_feature_importance(prettified=True)

labels = eval_pool.get_label()
s = 0
good = 0
bad = 0
for i in xrange(0,len(preds_raw)):
    v = -10 * preds_raw[i][0] - 0.1 * preds_raw[i][1] + 0.1 * preds_raw[i][2] + 0.5 * preds_raw[i][3]
    if v > 0:
        p = 1.
        good += 1
    else:
        p = -1.
        bad += 1
    s += p * float(labels[i])

print s/len(preds_raw), len(preds_raw)
print good, bad

#print eval_pool.get_label()[0:10]
#print(preds_raw[0:10])
#print(preds_class[0:10])
#print([2./(1+math.exp(-x))-1. for x in preds_raw])
