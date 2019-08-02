import numpy as np
import scipy.stats
from ndb import NDB

"""
Suppose p and q are two distributions on R^d. We calculate their distance.
"""


def sample_from(p, n_sample=1000):
    # p is tuple of (pi, mu, sigma)
    pi, mu, cov = p
    pi, mu, cov = np.array(pi), np.array(mu), np.array(cov)
    K, dim = mu.shape
    assert pi.shape == (K,)
    assert cov.shape == (K, dim, dim)

    samples = np.array([])
    counts = [0 for _ in range(K)]
    for _ in range(n_sample):
        idx = np.random.choice(range(K), p=pi)
        counts[idx] += 1
    for idx in range(K):
        # sample from (mu[idx], cov[idx])
        if not len(samples):
            samples = np.random.multivariate_normal(mu[idx], cov[idx], counts[idx])
        else:
            samples = np.vstack([samples,
                                np.random.multivariate_normal(mu[idx], cov[idx], counts[idx])])
    assert samples.shape == (n_sample, dim)
    return samples


def calc_NDB(p, q, n_sample=1000):
    # lower better
    train_samples, test_samples = sample_from(p, n_sample=n_sample), sample_from(q, n_sample=n_sample)
    ndb = NDB(training_data=train_samples)
    results = ndb.evaluate(test_samples)
    return results


def calc_negLogLikelihood(p, q, n_sample=1000):
    # lower better
    train_samples = sample_from(p, n_sample=n_sample)

    def pdf(x, pi, mu, cov):
        pi, mu, cov = np.array(pi), np.array(mu), np.array(cov)
        K, dim = mu.shape
        assert pi.shape == (K,)
        assert cov.shape == (K, dim, dim)

        return sum([pi[i] * np.exp(-0.5 * (x-mu[i]).T.dot(np.linalg.inv(cov[i])).dot(x-mu[i]))
                    / np.sqrt((2*np.pi) ** dim * np.linalg.det(cov[i]))
                    for i in range(K)])

    pi, mu, cov = q
    pi, mu, cov = np.array(pi), np.array(mu), np.array(cov)
    return -sum([np.log(pdf(x, pi, mu, cov)) for x in train_samples]) / n_sample


def calc_negELBO(p, q):
    pass


def calc_Wasserstein(p, q):
    pass


p = ([0.4, 0.6],
     [[-1.0, -1.0], [1.0, 1.0]],
     [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]],)
q = ([0.6, 0.4],
     [[-1.0, -1.0], [1.0, 1.0]],
     [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]],)
#print(calc_negLogLikelihood(p, q, n_sample=10000))