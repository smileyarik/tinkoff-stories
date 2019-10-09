awk 'NR>1{s=$2*-10+$3*-0.1+$4*0.1+$5*0.5; if(s>0)print 1;else print -1}' yareg/vl_output_001.txt > yareg_vl_score.txt
awk '{print $3}' yareg/vl_features_001.txt > weights.txt
awk '{print $3}' yareg/tr_features_001.txt > weights_train.txt

yareg=`paste yareg_vl_score.txt weights.txt | awk '{s+=$1*$2; w += $2 > 0 ? $2 : - $2}END{print s / w}'`

function print_score {
    what=$1
    awk '{print ($1 > 0 ? 1 : -1)}' < predict.$what > tmp
    score=`paste tmp weights.txt | awk -v yareg=$yareg '{if($1>1)$1=1; if($1<-1)$1=-1; s+=$1*$2; w += $2 > 0 ? $2 : - $2}END{print s / w " (" s / (w * yareg) ")"}'`
    echo $what = $score

    # just try and see optimal threshold
    # paste predict.$what weights.txt | LC_ALL=C sort +0g -1 > tmp2
    # all_pos_score=`awk '{s += $2}END{print s}' tmp2`
    # optimal_score=`awk -vscore=$all_pos_score -v yareg=$yareg '{score -= 2*$2; if (score>best_score){best_threshold=$1; best_score=score}; w+= $2 > 0 ? $2 : - $2}END{print best_score/w "(" (best_score / (w * yareg)) "), thr = " best_threshold ""}' tmp2`
    # echo "   opt $what = $optimal_score"
}

function eval {
    what=$1
    bash catboost.sh calc -m $what.catboost --input-path validate.txt --cd cd.txt --output-path tmp > /dev/null
    awk 'NR>1{print $2}' tmp > predict.$what
    print_score $what
}

#eval exp
# eval bblr
#eval hinge
eval classif
eval sigm0
#eval sigm0m1
#eval sigm0m2
#eval sigm0.5
eval sigm1
eval sigm2
#eval sigm3
#eval sigm3m1
#eval sigm3m2
eval sigm5
#eval sigm5m1
#eval sigm5m2
#eval sigm7.5
#eval sigm7.5m1
#eval sigm7.5m2
#eval sigm10

bash catboost.sh calc -m multi.catboost --input-path validate.txt --cd cd.txt --output-path tmp --prediction-type Probability > /dev/null
awk 'NR>1{s=$2*-10+$3*-0.1+$4*0.1+$5*0.5; if(s>0)print 1;else print -1}' tmp > predict.multi
print_score multi

bash catboost.sh calc -m multi-nw.catboost --input-path validate.txt --cd cd.txt --output-path tmp --prediction-type Probability > /dev/null
awk 'NR>1{s=$4+$5-$2-$3; if(s>0)print 1;else print -1}' tmp > predict.multi-nw
print_score multi-nw

#bash catboost.sh calc -m multi_patched.catboost --input-path validate.txt --cd cd.txt --output-path tmp --prediction-type Probability > /dev/null
#awk 'NR>1{s=$2*-10+$3*-0.1+$4*0.1+$5*0.5; if(s>0)print 1;else print -1}' tmp > predict.multi_patched
#print_score multi_patched

#paste predict.classif predict.sigm0 predict.sigm0.5 predict.sigm1 predict.sigm2 predict.sigm3 |
#   awk '{s=0;for(i=1;i<=NF;++i)s+=$i;print (s > 0 ? 1 : -1)}' > predict.ensemble
#print_score ensemble
