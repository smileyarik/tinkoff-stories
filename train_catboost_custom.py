import time
import random
import sys
import math
from six.moves import xrange
from catboost import Pool, CatBoostClassifier

class LoglossObjective(object):
    def calc_ders_range(self, approxes, targets, weights):
        # approxes, targets, weights are indexed containers of floats
        # (containers with only __len__ and __getitem__ defined).
        # weights parameter can be None.
        # Returns list of pairs (der1, der2)
        assert len(approxes) == len(targets)
        if weights is not None:
            assert len(weights) == len(approxes)

        exponents = []
        for index in xrange(len(approxes)):
            exponents.append(math.exp(approxes[index]))

        result = []
        for index in xrange(len(targets)):
            p = exponents[index] / (1 + exponents[index])**2
            der1 = 2 * p * targets[index]
            der2 = -2 * p * (exponents[index] - 1) * targets[index] / (exponents[index] + 1)
            #if approxes[index] != 0.0:
            #    time.sleep(random.random() * 5.)
            #    print approxes[index], targets[index], der1, der2

            if weights is not None:
                der1 *= weights[index]
                der2 *= weights[index]

            result.append((der1, der2))

        return result

    def get_final_error(self, error, weight):
        return error / (weight + 1e-38)

    def is_max_optimal(self):
        return True

    def evaluate(self, approxes, target, weight):
        # approxes is list of indexed containers
        # (containers with only __len__ and __getitem__ defined), one container
        # per approx dimension. Each container contains floats.
        # weight is one dimensional indexed container.
        # target is float.   
        # weight parameter can be None.
        # Returns pair (error, weights sum)

        assert len(approxes) == 1
        assert len(target) == len(approxes[0])

        approx = approxes[0]

        error_sum = 0.0
        weight_sum = 0.0

        for i in xrange(len(approx)):
            w = 1.0 if weight is None else weight[i]
            weight_sum += w
            error_sum += w * target[i] * (2. / (1 + math.exp(-approx[i])) - 1.)

        return error_sum, weight_sum



train_pool = Pool(data=sys.argv[1], column_description='cd')
eval_pool = Pool(data=sys.argv[2], column_description='cd')

# Initialize CatBoostClassifier with custom `loss_function`
model = CatBoostClassifier(loss_function=LoglossObjective(),
                           eval_metric=LoglossObjective(),
                           iterations=int(sys.argv[4]),
                           ignored_features=xrange(12,42))

# Fit model
model.fit(train_pool, eval_set=eval_pool)
# Only prediction_type='RawFormulVal' allowed with custom `loss_function`
preds_raw = model.predict(eval_pool,
                          prediction_type='RawFormulaVal')

model.save_model(sys.argv[3])

labels = eval_pool.get_label()
s = 0
for i in xrange(0,len(preds_raw)):
    p = 2./(1+math.exp(-preds_raw[i]))-1.
    s += p * float(labels[i])

print s/len(preds_raw), len(preds_raw)

#print eval_pool.get_label()[0:10]
#print(preds_raw[0:10])
#print(preds_class[0:10])
#print([2./(1+math.exp(-x))-1. for x in preds_raw])
