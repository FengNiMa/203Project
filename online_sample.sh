#!/bin/sh

A='2d_synthetic'
B='MoG5_full'
# oneline EM training
ONLINE_RESULT_PATH='online/'${A}'/penalized'
# full penalized EM
FULL_RESULT_PATH='online/'${A}'/full'
# dataset
DATASET_NAME=$A'/'$B
DATASET=${DATASET_NAME}'.npz'
# parameters
K=5
LAM=0
BATCH_SIZE=4

#python3 visualize.py $DATASET  --save_path ${DATASET_NAME}/dataset.png
#
## ------------------------------Standard EM------------------------------
## Standard EM training without adversaries, results-0.p
#python3 em_train.py --algo standard --output_dir results/${ONLINE_RESULT_PATH}\
#  --index 0\
#  --dataset_file $DATASET\
#  --k $K\
#  --lam $LAM
#echo 'Standard EM finished'

python3 visualize.py ${ONLINE_RESULT_PATH}/results-0.p  --save_path ${ONLINE_RESULT_PATH}/penalized-0.png\
  --plot_process True

# ------------------------------Online penEM------------------------------
# Construct online datasets
python3 construct_online_datasets.py --dataset_file $DATASET\
  --batch_size $BATCH_SIZE
echo 'Online data construction finished'

# do the first one, not from EM result
DATASET_this=${DATASET_NAME}1.npz
python3 em_train.py --algo penalized --output_dir results/${ONLINE_RESULT_PATH} \
  --index 1\
  --dataset_file $DATASET_this \
  --k $K\
  --lam $LAM
echo 'Online experiment 1 finished'

python3 visualize.py ${ONLINE_RESULT_PATH}/results-1.p --save_path ${ONLINE_RESULT_PATH}/penalized-1.png
echo 'Visualization 1 finished'

# Load previous result and run online penaized algorithm, results-i.p
i=2
while [ $i -lt $((7/$BATCH_SIZE+2)) ]
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

# ------------------------------Full penEM------------------------------
# Full penalized EM, results.p
python3 em_train.py --algo penalized --output_dir results/$FULL_RESULT_PATH \
  --dataset_file $DATASET\
  --k $K\
  --lam $LAM
echo 'Full penalized EM finished'

python3 visualize.py ${FULL_RESULT_PATH}/results.p --save_path ${FULL_RESULT_PATH}/full.png\
  --plot_process True

python3 metrics.py --dataset $DATASET\
  --result_path 'online/'${A}