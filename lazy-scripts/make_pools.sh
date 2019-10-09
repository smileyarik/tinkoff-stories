ver=003

function conv {
    awk -F'\t' -vOFS='\t' '{s=$3>0?1:-1;$2=$3*s;$3=s;print}'
}

function conv2 {
    awk -F'\t' -vOFS='\t' '{s=$3>0?1:0;$2=$3>0?$3:-$3;$3=s;print}'
}

function conv_multi {
    awk -F'\t' -vOFS='\t' '{$2=$3>0?$3:-$3;print}'
}

ver=$ver

conv < proper/data/tr_features_$ver.txt > train.txt
conv < proper/data/vl_features_$ver.txt > validate.txt


conv2 < proper/data/tr_features_$ver.txt > train_classif.txt
conv2 < proper/data/vl_features_$ver.txt > validate_classif.txt


conv_multi < proper/data/tr_features_$ver.txt > train_multi.txt
conv_multi < proper/data/vl_features_$ver.txt > validate_multi.txt