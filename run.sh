lstmver="001"
ver="001"

python split.py ../src_data/stories_reaction_train.csv ../data/tr_st_reactions.csv ../data/tr_tg_reactions.csv ../data/vl_st_reactions.csv ../data/vl_tg_reactions.csv ../data/tt_st_reactions.csv "2018-07-01 00:00:00" "2018-07-16 00:00:00" 2
python split.py ../src_data/transactions.csv ../data/tr_st_transactions.csv ../data/tr_tg_transactions.csv ../data/vl_st_transactions.csv ../data/vl_tg_transactions.csv ../data/tt_st_transactions.csv "7,1" "7,16" 1,2


python make_profiles.py ../src_data/stories_description.csv ../data/tr_st_reactions.csv ../data/tr_st_transactions.csv ../src_data/customer_test.csv ../data/tr_user_pickle.bin ../data/tr_item_pickle.bin
python make_profiles.py ../src_data/stories_description.csv ../data/vl_st_reactions.csv ../data/vl_st_transactions.csv ../src_data/customer_test.csv ../data/vl_user_pickle.bin ../data/vl_item_pickle.bin
python make_profiles.py ../src_data/stories_description.csv ../data/tt_st_reactions.csv ../data/tt_st_transactions.csv ../src_data/customer_test.csv ../data/tt_user_pickle.bin ../data/tt_item_pickle.bin

python make_features.py ../data/tr_user_pickle.bin ../data/tr_item_pickle.bin train "2018-07-01 00:00:00" ../data/tr_tg_reactions.csv ../data/tr_features_$ver.txt
python make_features.py ../data/vl_user_pickle.bin ../data/vl_item_pickle.bin train "2018-07-16 00:00:00" ../data/vl_tg_reactions.csv ../data/vl_features_$ver.txt
python make_features.py ../data/tt_user_pickle.bin ../data/tt_item_pickle.bin testt "2018-08-01 00:00:00" ../src_data/stories_reaction_test.csv ../data/tt_features_$ver.txt

at ../data/vl_features_$ver.txt > ../data/trvl_features_$ver.txt
#cat ../data/tr_features_$ver.txt >> ../data/trvl_features_$ver.txt

python train_catboost.py ../data/tr_features_$ver.txt ../data/vl_features_$ver.txt ../data/val_mmodel_$ver.cbm 1000
python train_catboost.py ../data/trvl_features_$ver.txt ../data/trvl_features_$ver.txt ../data/model_$ver.cbm 1000
#python train_catboost_custom.py ../data/tr_features_$ver.txt ../data/vl_features_$ver.txt ../data/val_cmodel_$ver.cbm 100

./catboost calc -m ../data/model_$ver.cbm --cd cd -o ../data/tt_output_$ver.txt --input-path ../data/tt_features_$ver.txt --prediction-type Probability
#./catboost calc -m ../data/val_model_$ver.cbm --cd cd -o ../data/dbg_tt_output_$ver.txt --input-path ../data/tt_features_$ver.txt
#./catboost calc -m ../data/val_model_$ver.cbm --cd cd -o ../data/vl_output_$ver.txt --input-path ../data/vl_features_$ver.txt --prediction-type Probability
#./catboost calc -m ../data/val_model_$ver.cbm --cd cd -o ../data/vl_output_$ver.txt --input-path ../data/vl_features_$ver.txt

cat ../data/tt_output_$ver.txt | awk -vOFS=',' 'BEGIN{print "answer_id,score"}NR>1{v=-10*$2-0.1*$3+0.1*$4+0.5*$5;if(v>0){p=1}else{p=-1}print $1,p}' > ../data/result_$ver.csv


