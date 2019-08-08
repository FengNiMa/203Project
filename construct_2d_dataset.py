import numpy as np
import math

def MoG_generation(N=1000):
    K = 5
    data_dim = 2

    pi = np.array([5/21, 5/21, 1/21, 5/21, 5/21])
    mu_list = [(2, 2), (2, 6), (4, 4), (6, 2), (6, 6)]
    mu = np.array(mu_list)
    cov = np.array([sigma*np.identity(data_dim) for sigma in [0.5,0.5,0.1,0.5,0.5]])

    adv_l = [(2, 4), (4, 2), (4, 6), (6, 4), (4.8, 4.8), (3.2, 4.8), (4.8, 3.2), (3.2, 3.2)]
    adv_sample = np.array(adv_l)

    samples_list = []
    while len(samples_list) < N:
        cls = np.random.randint(K)
        sample = np.random.multivariate_normal(mu[cls], cov[cls])

        dist = sample-[4,4]
        if 3 <= np.linalg.norm(dist, ord=1) <= 4 and dist[0]*dist[1] < 0:
            pass  # delete sample
        else:
            samples_list.append(sample)
    samples = np.array(samples_list)

    return pi, mu, cov, samples, adv_sample

pi, mu, cov, samples, adv_sample = MoG_generation()
data_fname = 'datasets/2d_synthetic/MoG5_full.npz'
np.savez(data_fname, pi=pi, mu=mu, cov=cov, samples=samples, adv_sample=adv_sample)