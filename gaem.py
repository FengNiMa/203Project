import os, sys, time, pickle,math, argparse, json, torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits

from algorithm import *

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=2)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=2)
        self.fc1 = nn.Linear(20, 30)
        self.fc2 = nn.Linear(30, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 20)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x)

def oracle_test(oracle, generated_data):
    gen_data = torch.Tensor([[x.reshape((8, 8))] for x in generated_data])
    latent_gen = network(gen_data).detach().numpy()
    count = 0
    adv_sample_index = []
    for i,x in enumerate(latent_gen):
        if max(scipy.special.softmax(x)) < 0.90:
            count += 1
            adv_sample_index.append(i)
            #plot_one_digit(digits_new[i])
    did_not_pass = count/float(len(ar))
    print("Percentage classified as adversary: %.2f" % did_not_pass)
    return np.array(adv_sample_index), did_not_pass

def plot_digits(data, chang=10, kuan=10):
    fig, ax = plt.subplots(chang, kuan, figsize=(8, 8),
                           subplot_kw=dict(xticks=[], yticks=[]))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    for i, axi in enumerate(ax.flat):
        #data_thresh = np.array([0 if xi < 4 else xi for xi in data[i]])
        im = axi.imshow(data[i].reshape(8, 8), cmap='binary')
        im.set_clim(0, 16)
    return fig


def main():


    # set up output
    '''
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir,'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    logger = Logger(os.path.join(args.output_dir,'log.txt'))
    logger.activate()
    '''

    #load data and oracle
    digits = load_digits()
    original_data = digits.data #dim = 64
    print("data loaded.")
    og = plot_digits(digits.data)
    og.savefig('./figures/mnist/original.png', bbox_inches='tight')
    pca = PCA(0.99, whiten=True) 
    X = pca.fit_transform(digits.data) #dim = 41
    np.savez('./datasets/MNIST/mnist_1797.npz', data=X)

    #em baseline
    
    pis, mus, sigmas = init_guess(10, 41)
    pi, mu, cov, losses, iter_n = em_gmm(X, pis, mus, sigmas)
    with open( './figures/mnist/baselinemodel.p', 'wb') as p:
        results = {"init":[pis, mus, sigmas],
            "pi":pi, "mu":mu, "cov":cov,
            "loss":losses, "iters":iter_n}
        pickle.dump(results, p)

    '''
    results = pickle.load(open( './figures/mnist/baselinemodel.p', 'rb'))
    pi, mu, cov, losses = results['pi'], results['mu'], results['conv'], results['loss'] 
    '''
    generated_samples_pca = sample(100, pi, mu, cov)

    generated_samples = pca.inverse_transform(generated_samples_pca)
    basline_fig = plot_digits(generated_samples)
    baseline_fig.savefig('./figures/mnist/baseline.png', bbox_inches='tight')
    print("Baseline loss: %0.2f" % losses[-1])


    oracle = Net().load_state_dict("./notebooks/model95.pth")
    epochs = 10
    Z = np.array([])

    for i in range(1, epochs+1):
        # query oracle to get adv samples with new z
        adv_sample_index, _ = oracle_test(oracle, generated_samples)
        Z = np.concatenate((Z, generated_samples_pca[adv_sample_index]))

        # penalized em
        pi, mu, cov, p_loss, _, _ =  em_gmm_penalized(X, Z, pi, mu, cov, lmda=10)
        generated_samples_pca = sample(100, pi, mu, cov)
        generated_samples = pca.inverse_transform(baseline_samples_pca)
        epoch_fig = plot_digits(generated_samples)
        epoch_fig.savefig('./figures/mnist/epoch_%d.png'%i, bbox_inches='tight')
        np.savez('./datasets/MNIST/z_%d.npz'%i, data=Z)
        print("Epoch %d loss: %0.2f" % (i,p_loss))

if __name__ == '__main__':
    main()
