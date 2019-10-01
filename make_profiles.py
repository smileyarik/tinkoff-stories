import datetime
import sys
import csv
import json
from profiles import *
from collections import defaultdict
import pickle

#python make_profiles.py ../src_data/stories_description.csv ../data/tr_st_reactions.csv ../data/tr_st_transactions.csv ../src_data/customer_test.csv ../data/tr_user_pickle.bin ../data/tr_item_pickle.bin

def make_counters():
    return Counters()

if __name__ == '__main__':
    item_counters = defaultdict(make_counters)
    user_counters = defaultdict(make_counters)

    print "Read user catalogue"
    with open(sys.argv[4]) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader, None)
        for row in reader:
            #customer_id,product_0,product_1,product_2,product_3,product_4,product_5,product_6,gender_cd,age,marital_status_cd,children_cnt,first_session_dttm,job_position_cd,job_title
            user_id = int(row[0])
            user = user_counters[user_id]
            for i in xrange(1,8):
                user.set(OT_PRODUCT, CT_HAS, RT_SUM, (i-1, row[i]), 1, 0)
            user.set(OT_GENDER, CT_HAS, RT_SUM, row[8], 1, 0)
            user.set(OT_AGE, CT_HAS, RT_SUM, int(float(row[9])) if row[9] != '' else 0, 1, 0)
            user.set(OT_MARITAL, CT_HAS, RT_SUM, row[10], 1, 0)
            user.set(OT_CHILDREN, CT_VALUE, RT_SUM, '', int(float(row[11])) if row[11] != '' else 0, 0)
            user.set(OT_JOB, CT_HAS, RT_SUM, row[13], 1, 0)

    # READ ITEM CATALOGUE #
    print "Reading catalogue"
    targ = defaultdict(lambda:set())
    with open(sys.argv[1]) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader, None)
        for row in reader:
            item_id = int(row[0])
            item = item_counters[item_id]


    event_dict = {'view' : CT_VIEW, 'like' : CT_LIKE, 'dislike' : CT_DISLIKE, 'skip' : CT_SKIP}

    # PARSE TRANSACTIONS #
    print "Parsing transactions"
    with open(sys.argv[3]) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        # customer_id,transaction_month,transaction_day,transaction_amt,merchant_id,merchant_mcc
        count = 1
        for row in reader:
            if count % 100000 == 0:
                print count
            count += 1
            user_id = int(row[0])
            ts = int(datetime.datetime.strptime("2018-%2.2d-%2.2d 00:00:00" % (int(row[1]), int(row[2])), "%Y-%m-%d %H:%M:%S").strftime("%s"))
            user.add(OT_MCC, CT_TRANSACTION, RT_SUM, row[5], float(row[3]), ts)

    # PARSE REACTIONS
    print "Parsing reactions"
    with open(sys.argv[2]) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        # customer_id,story_id,event_dttm,event
        count = 1
        for row in reader:
            if count % 100000 == 0:
                print count
            count += 1

            user_id = int(row[0])
            item_id = int(row[1])
            event = event_dict[row[3]]
            user = user_counters[user_id]
            item = item_counters[item_id]
            ts = int(datetime.datetime.strptime(row[2], "%Y-%m-%d %H:%M:%S").strftime("%s"))

            for rt in [RT_SUM, RT_7D, RT_30D]:
                item.add(OT_GLOBAL, CT_SHOW, rt, '', 1, ts)
                item.add(OT_GLOBAL, event, rt, '', 1, ts)
                user.add(OT_GLOBAL, CT_SHOW, rt, '', 1, ts)
                user.add(OT_GLOBAL, event, rt, '', 1, ts)

                for e in [event, CT_SHOW]:
                    item.add(OT_GLOBAL, e, rt, '', 1, ts)
                    user.add(OT_GLOBAL, e, rt, '', 1, ts)

                    item.update_from(user, OT_PRODUCT, CT_HAS, RT_SUM, e, rt, ts)
                    item.update_from(user, OT_GENDER, CT_HAS, RT_SUM, e, rt, ts)
                    item.update_from(user, OT_AGE, CT_HAS, RT_SUM, e, rt, ts)
                    item.update_from(user, OT_MARITAL, CT_HAS, RT_SUM, e, rt, ts)
                    item.update_from(user, OT_CHILDREN, CT_VALUE, RT_SUM, e, rt, ts)
                    item.update_from(user, OT_JOB, CT_HAS, RT_SUM, e, rt, ts)

                    item.update_from(user, OT_MCC, CT_TRANSACTION, RT_SUM, e, rt, ts)

    print "Dumping user profiles"
    with open(sys.argv[5], 'w') as user_pickle:
        pickle.dump(user_counters, user_pickle)

    print "Dumping item profiles"
    with open(sys.argv[6], 'w') as item_pickle:
        pickle.dump(item_counters, item_pickle)


