what=full.classif

#cat train_classif.txt validate_classif.txt > full_classif.txt

#bash catboost.sh fit -m $what.catboost -f full_classif.txt --cd cd.txt -i 1500 -w 0.001 --loss-function "UserPerObjMetric:alpha=5"
#bash catboost.sh fit -m $what.catboost -t validate_classif.txt -f train_classif.txt --cd cd.txt $ignore -i 1000 -w 0.03 --loss-function Logloss
#bash catboost.sh calc -m $what.catboost --input-path proper/data/tt_features_002.txt --cd cd-noweights.txt --output-path tmp > /dev/null
#awk -vOFS=',' 'BEGIN{print "answer_id,score"} NR>1{print $1,($2 > 0 ? 1 : -1)}' tmp > submission.csv

what=full.multi-nw
cat train_multi.txt validate_multi.txt > full_multi.txt
bash catboost.sh fit -m $what.catboost -f full_multi.txt --cd cd.txt -i 1000 -w 0.01 --loss-function MultiClass
bash catboost.sh calc -m $what.catboost --input-path proper/data/tt_features_003.txt --cd cd.txt --output-path tmp --prediction-type Probability > /dev/null
awk -vOFS=',' 'BEGIN{print "answer_id,score"} NR>1{s=$4+$5-$2-$3; print $1, (s>0?1:-1)}' tmp > submission.csv
