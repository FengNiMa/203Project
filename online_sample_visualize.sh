#!/bin/sh

# oneline EM training
ONLINE_RESULT_PATH='online/2d_synthetic/penalized'
# full penalized EM
FULL_RESULT_PATH='online/2d_synthetic/full'
# dataset
DATASET_NAME='2d_synthetic/MoG5_full'
DATASET='2d_synthetic/MoG5_full.npz'

python visualize.py $DATASET  --save_path ${DATASET_NAME}/dataset.jpg
python visualize.py $FULL_RESULT_PATH/results.p  --save_path ${FULL_RESULT_PATH}/full.jpg

i=0
while [ $i -lt 9 ]
do
	python visualize.py ${ONLINE_RESULT_PATH}/results-${i}.p --save_path ${ONLINE_RESULT_PATH}/penalized-${i}.png 
	echo 'Visualization '$i' finished'
  	i=$(($i+1))
done