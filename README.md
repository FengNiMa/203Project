# 203Project
On the EM Algorithm for Gaussian Mixture Models with Adversarial Regularization

## Datasets index
- Multi-adv-0: Missing outside, 4 adv
- Multi-adv-1: Missing 4 strips ("#" shape), 4 adv
- Multi-adv-2: Missing 4 strips ("#" shape), 8 adv
- Multi-adv-3: Missing 2 strips ("+" shape), 8 adv

## Dependencies
Python 3, numpy, scipy, (and tqdm for logging)

## Test Run
# standard
```
python em_train.py --algo penalized --output_dir results/multi-adv-test --dataset_file multi-adv-0/data_multi_adv.npz
```
# online
```
./online_sample.sh
```

## Notice
- Don't modify the Results.ipynb directly; copy it to a new file named Results-xxx.ipynb if changing is needed (it will be in .gitignore).
- Please put datasets, figures and results into corresponding folders (refer to the datasets index).

