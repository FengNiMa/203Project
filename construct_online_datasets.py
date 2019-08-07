import numpy as np
import os, sys, time, pickle
import argparse, json

def parse():

    parser = argparse.ArgumentParser(description='construct online datasets parser')

    parser.add_argument('--dataset_file', type=str, default='2d_synthetic/MoG5_full.npz', help='full dataset file')

    return parser.parse_args()


def main():
    args = parse()
    data = np.load(os.path.join('datasets', args.dataset_file))
    train = data['samples']
    adv = data['adv_sample']
    index = list(range(adv.shape[0]))
    np.random.shuffle(index)
    adv = adv[index]
    for i in range(1, adv.shape[0]+1):
        np.savez(os.path.join('datasets',
                              args.dataset_file.split('.')[0] + str(i) + '.' + args.dataset_file.split('.')[1]),
                 samples=train,
                 adv_sample=adv[:i])


if __name__ == '__main__':
    main()
