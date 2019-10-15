ver="001"

python split.py ../src_data/stories_reaction_train.csv ../data/tr_st_reactions.csv ../data/tr_tg_reactions.csv ../data/vl_st_reactions.csv ../data/vl_tg_reactions.csv ../data/tt_st_reactions.csv "2018-07-01 00:00:00" "2018-07-16 00:00:00" 2
python split.py ../src_data/transactions.csv ../data/tr_st_transactions.csv ../data/tr_tg_transactions.csv ../data/vl_st_transactions.csv ../data/vl_tg_transactions.csv ../data/tt_st_transactions.csv "7,1" "7,16" 1,2

cat ../src_data/stories_reaction_test.csv ../src_data/stories_reaction_train.csv | awk -F ',' '{print $2}' | sort -n | uniq | grep -v story > ../data/all_story_id.txt

# WE ALWAYS USE tt_st_transactions intentionaly
python make_profiles.py ../src_data/stories_description.csv ../data/tr_st_reactions.csv ../data/tt_st_transactions.csv ../src_data/customer_test.csv ../data/tr_user_pickle.bin ../data/tr_item_pickle.bin
python make_profiles.py ../src_data/stories_description.csv ../data/vl_st_reactions.csv ../data/tt_st_transactions.csv ../src_data/customer_test.csv ../data/vl_user_pickle.bin ../data/vl_item_pickle.bin
python make_profiles.py ../src_data/stories_description.csv ../data/tt_st_reactions.csv ../data/tt_st_transactions.csv ../src_data/customer_test.csv ../data/tt_user_pickle.bin ../data/tt_item_pickle.bin

export OPENBLAS_NUM_THREADS=1;
python learn_nn2.py ../data/tr_st_reactions.csv ../data/tr_tg_reactions.csv ../src_data/customer_test.csv ../data/all_story_id.txt ../data/tr_model_002.bin
python learn_nn2.py ../data/vl_st_reactions.csv ../data/vl_tg_reactions.csv ../src_data/customer_test.csv ../data/all_story_id.txt ../data/vl_model_002.bin
python learn_nn2.py ../data/tt_st_reactions.csv ../data/vl_tg_reactions.csv ../src_data/customer_test.csv ../data/all_story_id.txt ../data/tt_model_002.bin

python learn_nn.py ../data/tr_st_reactions.csv ../data/tr_tg_reactions.csv ../src_data/customer_test.csv ../data/all_story_id.txt ../data/tr_model_001.bin
python learn_nn.py ../data/vl_st_reactions.csv ../data/vl_tg_reactions.csv ../src_data/customer_test.csv ../data/all_story_id.txt ../data/vl_model_001.bin
python learn_nn.py ../data/tt_st_reactions.csv ../data/vl_tg_reactions.csv ../src_data/customer_test.csv ../data/all_story_id.txt ../data/tt_model_001.bin

python make_features.py ../data/tr_user_pickle.bin ../data/tr_item_pickle.bin train "2018-07-01 00:00:00" ../data/tr_tg_reactions.csv ../data/tr_features_$ver.txt ../data/tr_model_001.bin ../src_data/customer_test.csv ../data/all_story_id.txt ../src_data/stories_description.csv ../data/tr_model_002.bin ../data/tr_model_lstm.bin ../data/tr_st_reactions.csv
python make_features.py ../data/vl_user_pickle.bin ../data/vl_item_pickle.bin train "2018-07-16 00:00:00" ../data/vl_tg_reactions.csv ../data/vl_features_$ver.txt ../data/vl_model_001.bin ../src_data/customer_test.csv ../data/all_story_id.txt ../src_data/stories_description.csv ../data/vl_model_002.bin ../data/vl_model_lstm.bin ../data/vl_st_reactions.csv
python make_features.py ../data/tt_user_pickle.bin ../data/tt_item_pickle.bin testt "2018-08-01 00:00:00" ../src_data/stories_reaction_test.csv ../data/tt_features_$ver.txt ../data/tt_model_001.bin ../src_data/customer_test.csv ../data/all_story_id.txt ../src_data/stories_description.csv ../data/tt_model_002.bin ../data/tt_model_lstm.bin ../data/tt_st_reactions.csv

tail -77000 ../data/vl_features_$ver.txt > ../data/vl_features_short.txt

python train_catboost.py ../data/vl_features_short.txt ../data/vl_features_short.txt ../data/model_$ver.cbm 600

./catboost calc -m ../data/model_$ver.cbm --cd cd -o ../data/tt_output_$ver.txt --input-path ../data/tt_features_$ver.txt --prediction-type Probability

cat ../data/tt_output_$ver.txt | awk -vOFS=',' 'BEGIN{print "answer_id,score"}NR>1{v=-10*$2-0.1*$3+0.1*$4+0.5*$5;if(v>0){p=1}else{p=-1}print $1,p}' > ../data/result_$ver.csv

