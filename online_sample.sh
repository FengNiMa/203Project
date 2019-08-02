#!/bin/sh

ONLINE_RESULT_PATH='results/online/multi-adv-0/standard'
FINAL_RESULT_PATH='results/online/multi-adv-0/penalized'
DATASET='multi-adv-0/data_multi_adv.npz'

# Standard em training without adversaries
python em_train.py --algo standard --output_dir $ONLINE_RESULT_PATH \
--dataset_file $DATASET

# Load previous result and run penaized algorithm
python em_train.py --algo penalized --output_dir $FINAL_RESULT_PATH \
 --dataset_file $DATASET \
 --load_path ${ONLINE_RESULT_PATH}/results.p