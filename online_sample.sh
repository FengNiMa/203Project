#!/bin/sh

# oneline EM training
ONLINE_RESULT_PATH='results/online/2d_synthetic/penalized'
# full penalized EM
FULL_RESULT_PATH='results/online/2d_synthetic/full'
# dataset
DATASET_NAME='2d_synthetic/MoG5_full'
DATASET='2d_synthetic/MoG5_full.npz'

# Standard em training without adversaries
python3 em_train.py --algo standard --output_dir ${ONLINE_RESULT_PATH}\
  --index 0\
  --dataset_file $DATASET
echo 'Standard EM finished'

# Construct online datasets
python3 construct_online_datasets.py --dataset_file $DATASET
echo 'Online data construction finished'

# Load previous result and run penaized algorithm
a=1
while [ $a -lt 9 ]
do
  PREV_A=$(($a-1))
  DATASET_this=$DATASET_NAME$a.npz
  python3 em_train.py --algo penalized --output_dir ${ONLINE_RESULT_PATH} \
    --index $a\
    --dataset_file $DATASET_this \
    --load_path ${ONLINE_RESULT_PATH}/results-${PREV_A}.p
  ## TO DO: save figs, calculate metrics
  echo 'Online experiment '$a' finished'
  a=$(($a+1))
done

# Full penalized EM
python3 em_train.py --algo penalized --output_dir $FULL_RESULT_PATH \
  --dataset_file $DATASET
echo 'Full penalized EM finished'