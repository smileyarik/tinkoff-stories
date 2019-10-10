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

def date_to_timestamp(strdate):
    # windows doesn't support "%s", so let's use explicit calculation for timestamp
    epoch = datetime.datetime(1970,1,1)
    event_time = datetime.datetime.strptime(strdate, "%Y-%m-%d %H:%M:%S")
    return int((event_time - epoch).total_seconds())

if __name__ == '__main__':
    item_counters = defaultdict(make_counters)
    user_counters = defaultdict(make_counters)

    print("Read user catalogue")
    with open(sys.argv[4], encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader, None)
        for row in reader:
            #customer_id,product_0,product_1,product_2,product_3,product_4,product_5,product_6,gender_cd,age,marital_status_cd,children_cnt,first_session_dttm,job_position_cd,job_title
            user_id = int(row[0])
            user = user_counters[user_id]
            for i in range(1,8):
                user.set(OT_PRODUCT, CT_HAS, RT_SUM, (i-1, row[i]), 1, 0)
            user.set(OT_GENDER, CT_HAS, RT_SUM, row[8], 1, 0)
            user.set(OT_AGE, CT_HAS, RT_SUM, int(float(row[9])) if row[9] != '' else 0, 1, 0)
            user.set(OT_MARITAL, CT_HAS, RT_SUM, row[10], 1, 0)
            user.set(OT_CHILDREN, CT_VALUE, RT_SUM, '', int(float(row[11])) if row[11] != '' else 0, 0)
            user.set(OT_JOB, CT_HAS, RT_SUM, row[13], 1, 0)

    # READ ITEM CATALOGUE #
    print("Reading catalogue")
    targ = defaultdict(lambda:set())
    with open(sys.argv[1], encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader, None)
        for row in reader:
            item_id = int(row[0])
            item = item_counters[item_id]


    event_dict = {'view' : CT_VIEW, 'like' : CT_LIKE, 'dislike' : CT_DISLIKE, 'skip' : CT_SKIP}

    # PARSE TRANSACTIONS #
    print("Parsing transactions")
    with open(sys.argv[3], encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        # customer_id,transaction_month,transaction_day,transaction_amt,merchant_id,merchant_mcc
        count = 1
        for row in reader:
            if (len(row)) == 0: continue # stupid windows
            if count % 100000 == 0:
                print(count)
            count += 1
            user_id = int(row[0])
            user = user_counters[user_id]
            month, day = int(row[1]), int(row[2])
            ts = date_to_timestamp("2018-%2.2d-%2.2d 00:00:00" % (month, day))
            amount = float(row[3]) + 250 # transaction amounts are rounded down with precision of 500
            mcc = row[5]
            user.add(OT_MCC, CT_TRANSACTION_AMOUNT, RT_SUM, mcc, amount, ts)
            user.add(OT_GLOBAL, CT_TRANSACTION_COUNT, RT_SUM, '', 1, ts)
            user.add(OT_GLOBAL, CT_TRANSACTION_DATE, RT_SUM, "%2d-%2d" % (month, day), 1, ts)
            user.add(OT_GLOBAL, CT_TRANSACTION_AMOUNT, RT_SUM, '', amount, ts)

        for user_id, user in user_counters.items():
            total_amount = user.get(OT_GLOBAL, CT_TRANSACTION_AMOUNT, RT_SUM, '', 0)
            for mcc, amount in user.slice(OT_MCC, CT_TRANSACTION_AMOUNT, RT_SUM).items():
                user.add(OT_MCC, CT_TRANSACTION_AMOUNT_RATIO, RT_SUM, mcc, amount.get(0, RT_SUM) / total_amount, 0)

    # PARSE REACTIONS
    print("Parsing reactions")
    with open(sys.argv[2], encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        # customer_id,story_id,event_dttm,event
        count = 1
        for row in reader:
            if (len(row)) == 0: continue # stupid windows
            if count % 100000 == 0:
                print(count)
            count += 1

            user_id = int(row[0])
            item_id = int(row[1])
            event = event_dict[row[3]]
            user = user_counters[user_id]
            item = item_counters[item_id]
            ts = date_to_timestamp(row[2])

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

                    item.update_from(user, OT_MCC, CT_TRANSACTION_AMOUNT_RATIO, RT_SUM, e, rt, ts)

    print("Dumping user profiles")
    with open(sys.argv[5], 'wb') as user_pickle:
        pickle.dump(user_counters, user_pickle)

    print("Dumping item profiles")
    with open(sys.argv[6], 'wb') as item_pickle:
        pickle.dump(item_counters, item_pickle)


