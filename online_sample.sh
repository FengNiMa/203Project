#!/bin/sh

# oneline EM training
ONLINE_RESULT_PATH='online/2d_synthetic/penalized'
# full penalized EM
FULL_RESULT_PATH='online/2d_synthetic/full'
# dataset
DATASET_NAME='2d_synthetic/MoG5_full'
DATASET='2d_synthetic/MoG5_full.npz'
# parameters
K=5
LAM=800

# Standard em training without adversaries, results-0.p
python3 em_train.py --algo standard --output_dir results/${ONLINE_RESULT_PATH}\
  --index 0\
  --dataset_file $DATASET\
  --k $K\
  --lam $LAM
echo 'Standard EM finished'

python3 visualize.py $DATASET  --save_path ${DATASET_NAME}/dataset.jpg
python3 visualize.py ${ONLINE_RESULT_PATH}/results-0.p  --save_path ${ONLINE_RESULT_PATH}/penalized-0.jpg\
  --plot_process True

# Construct online datasets
python3 construct_online_datasets.py --dataset_file $DATASET
echo 'Online data construction finished'

# Load previous result and run penaized algorithm, results-i.p
i=1
while [ $i -lt 9 ]
do
  PREV_i=$(($i-1))
  DATASET_this=$DATASET_NAME$i.npz
  python3 em_train.py --algo penalized --output_dir results/${ONLINE_RESULT_PATH} \
    --index $i\
    --dataset_file $DATASET_this \
    --load_path results/${ONLINE_RESULT_PATH}/results-${PREV_i}.p\
    --k $K\
    --lam $LAM
  echo 'Online experiment '$i' finished'

  python3 visualize.py ${ONLINE_RESULT_PATH}/results-${i}.p --save_path ${ONLINE_RESULT_PATH}/penalized-${i}.png
	echo 'Visualization '$i' finished'

  ## TO DO: save figs, calculate metrics
  i=$(($i+1))
done

# Full penalized EM, results.p
python3 em_train.py --algo penalized --output_dir results/$FULL_RESULT_PATH \
  --dataset_file $DATASET\
  --k $K\
  --lam $LAM
echo 'Full penalized EM finished'

python3 visualize.py ${FULL_RESULT_PATH}/results.p --save_path ${FULL_RESULT_PATH}/full.png\
  --plot_process True
