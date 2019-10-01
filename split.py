import csv
import sys

train_ts = ','.join([f.zfill(30) for f in sys.argv[7].split(',')])
valid_ts = ','.join([f.zfill(30) for f in sys.argv[8].split(',')])
ts_f = [int(x) for x in sys.argv[9].split(',')]

with open(sys.argv[1]) as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    headers = next(reader, None)

    train_stat_f = open(sys.argv[2], 'w')
    train_stat_writer = csv.writer(train_stat_f, delimiter=',')
    train_targ_f = open(sys.argv[3], 'w')
    train_targ_writer = csv.writer(train_targ_f, delimiter=',')

    valid_stat_f = open(sys.argv[4], 'w')
    valid_stat_writer = csv.writer(valid_stat_f, delimiter=',')
    valid_targ_f = open(sys.argv[5], 'w')
    valid_targ_writer = csv.writer(valid_targ_f, delimiter=',')

    test_stat_f = open(sys.argv[6], 'w')
    test_stat_writer = csv.writer(test_stat_f, delimiter=',')

    sortedlist = sorted(reader, key=lambda row: ','.join([row[f].zfill(30) for f in ts_f]), reverse=False)
    for row in sortedlist:
        ts = ','.join([row[f].zfill(30) for f in ts_f])
        test_stat_writer.writerow(row)
        if ts < train_ts:
            train_stat_writer.writerow(row)
            valid_stat_writer.writerow(row)
        elif ts < valid_ts:
            valid_stat_writer.writerow(row)
            train_targ_writer.writerow(row)
        else:
            valid_targ_writer.writerow(row)
