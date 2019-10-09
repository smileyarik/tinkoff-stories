bash catboost.sh fit -m multi-nw.catboost -t validate_multi.txt -f train_multi.txt --cd cd.txt $ignore -i 1000 -w 0.01 --loss-function MultiClass

bash catboost.sh fit -m classif.catboost -t validate_classif.txt -f train_classif.txt --cd cd.txt $ignore -i 1000 -w 0.01 --loss-function Logloss
bash catboost.sh fit -m multi.catboost -t validate_multi.txt -f train_multi.txt --cd cd-noweights.txt $ignore -i 1000 -w 0.03 --loss-function MultiClass
bash catboost.sh fit -m multi-nw.catboost -t validate_multi.txt -f train_multi.txt --cd cd.txt $ignore -i 1000 -w 0.01 --loss-function MultiClass
bash catboost.sh fit -m sigm0.catboost -t validate_classif.txt -f train_classif.txt --cd cd.txt $ignore -i 1000 -w 0.01 --loss-function UserPerObjMetric:alpha=0
bash catboost.sh fit -m sigm1.catboost -t validate_classif.txt -f train_classif.txt --cd cd.txt $ignore -i 1000 -w 0.01 --loss-function UserPerObjMetric:alpha=1
bash catboost.sh fit -m sigm2.catboost -t validate_classif.txt -f train_classif.txt --cd cd.txt $ignore -i 1000 -w 0.01 --loss-function UserPerObjMetric:alpha=2
bash catboost.sh fit -m sigm5.catboost -t validate_classif.txt -f train_classif.txt --cd cd.txt $ignore -i 1000 -w 0.01 --loss-function UserPerObjMetric:alpha=5

exit

bash catboost.sh fit -m hinge.catboost -t validate.txt -f train.txt --cd cd.txt $ignore -i 1000 -w 0.01 --loss-function UserPerObjMetric:hinge=1 --leaf-estimation-method Gradient #&
bash catboost.sh fit -m classif.catboost -t validate_classif.txt -f train_classif.txt --cd cd.txt $ignore -i 1000 -w 0.03 --loss-function Logloss #&
bash catboost.sh fit -m sigm0.catboost -t validate_classif.txt -f train_classif.txt --cd cd.txt $ignore -i 1000 -w 0.01 --loss-function UserPerObjMetric:alpha=0 #&
bash catboost.sh fit -m sigm0.5.catboost -t validate_classif.txt -f train_classif.txt --cd cd.txt $ignore -i 1000 -w 0.01 --loss-function UserPerObjMetric:alpha=0.5 #&

wait

bash catboost.sh fit -m sigm1.catboost -t validate_classif.txt -f train_classif.txt --cd cd.txt $ignore -i 1000 -w 0.01 --loss-function UserPerObjMetric:alpha=1 #&
bash catboost.sh fit -m sigm2.catboost -t validate_classif.txt -f train_classif.txt --cd cd.txt $ignore -i 1000 -w 0.01 --loss-function UserPerObjMetric:alpha=2 #&
bash catboost.sh fit -m sigm3.catboost -t validate_classif.txt -f train_classif.txt --cd cd.txt $ignore -i 1000 -w 0.01 --loss-function UserPerObjMetric:alpha=3 #&
bash catboost.sh fit -m sigm5.catboost -t validate_classif.txt -f train_classif.txt --cd cd.txt $ignore -i 1000 -w 0.01 --loss-function UserPerObjMetric:alpha=5 #&

wait

bash catboost.sh fit -m sigm7.5.catboost -t validate_classif.txt -f train_classif.txt --cd cd.txt $ignore -i 1000 -w 0.01 --loss-function UserPerObjMetric:alpha=7.5 #&
bash catboost.sh fit -m sigm10.catboost -t validate_classif.txt -f train_classif.txt --cd cd.txt $ignore -i 1000 -w 0.01 --loss-function UserPerObjMetric:alpha=10 #&
bash catboost.sh fit -m multi.catboost -t validate_multi.txt -f train_multi.txt --cd cd-noweights.txt $ignore -i 1000 -w 0.03 --loss-function MultiClass #&
bash catboost.sh fit -m multi-nw.catboost -t validate_multi.txt -f train_multi.txt --cd cd.txt $ignore -i 1000 -w 0.01 --loss-function MultiClass

wait

# bash catboost.sh fit -m sigm0m1.catboost -t validate_classif.txt -f train_classif.txt --cd cd.txt $ignore -i 1000 -w 0.01 --loss-function "UserPerObjMetric:alpha=0;margin=1" #&
# bash catboost.sh fit -m sigm0m1.catboost -t validate_classif.txt -f train_classif.txt --cd cd.txt $ignore -i 1000 -w 0.01 --loss-function "UserPerObjMetric:alpha=0;margin=1" #&
# bash catboost.sh fit -m sigm3m1.catboost -t validate_classif.txt -f train_classif.txt --cd cd.txt $ignore -i 1000 -w 0.01 --loss-function "UserPerObjMetric:alpha=3;margin=1" #&
# bash catboost.sh fit -m sigm5m1.catboost -t validate_classif.txt -f train_classif.txt --cd cd.txt $ignore -i 1000 -w 0.01 --loss-function "UserPerObjMetric:alpha=5;margin=1" #&
# bash catboost.sh fit -m sigm7.5m1.catboost -t validate_classif.txt -f train_classif.txt --cd cd.txt $ignore -i 1000 -w 0.01 --loss-function "UserPerObjMetric:alpha=7.5;margin=1" #&


# bash catboost.sh fit -m sigm0m2.catboost -t validate_classif.txt -f train_classif.txt --cd cd.txt $ignore -i 1000 -w 0.01 --loss-function "UserPerObjMetric:alpha=0;margin=2" #&
# bash catboost.sh fit -m sigm3m2.catboost -t validate_classif.txt -f train_classif.txt --cd cd.txt $ignore -i 1000 -w 0.01 --loss-function "UserPerObjMetric:alpha=3;margin=2" #&
# bash catboost.sh fit -m sigm5m2.catboost -t validate_classif.txt -f train_classif.txt --cd cd.txt $ignore -i 1000 -w 0.01 --loss-function "UserPerObjMetric:alpha=5;margin=2" #&
# bash catboost.sh fit -m sigm7.5m2.catboost -t validate_classif.txt -f train_classif.txt --cd cd.txt $ignore -i 1000 -w 0.01 --loss-function "UserPerObjMetric:alpha=7.5;margin=2" #&

# wait
