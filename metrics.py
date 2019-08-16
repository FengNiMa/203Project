import numpy as np
import scipy.stats
from scipy.linalg import sqrtm
from ndb import NDB
import math, os, argparse, pickle
from visualize import MoG_prob_

"""
Suppose p and q are two distributions on R^d. We calculate their distance.
p = true, q = learned
"""

def parse():
    parser = argparse.ArgumentParser(description='distance comparison')
    parser.add_argument('--n_sample', type=int, default=1000, help='number of samples to estimate distance')
    parser.add_argument('--dataset', type=str, default='2d_synthetic/MoG5_full.npz', help='dataset')
    parser.add_argument('--result_path', type=str, default='online/2d_synthetic', help='result path')
    return parser.parse_args()

class distribution():
    def __init__(self, pi, mu, cov):
        self.pi = np.array(pi)
        self.mu = np.array(mu)
        self.cov = np.array(cov)

def sample_from(p, n_sample=1000):

    K, dim = p.mu.shape
    assert p.pi.shape == (K,)
    assert p.cov.shape == (K, dim, dim)

    samples = np.array([])
    counts = [0 for _ in range(K)]
    for _ in range(n_sample):
        idx = np.random.choice(range(K), p=p.pi)
        counts[idx] += 1
    for idx in range(K):
        # sample from (mu[idx], cov[idx])
        if not len(samples):
            samples = np.random.multivariate_normal(p.mu[idx], p.cov[idx], counts[idx])
        else:
            samples = np.vstack([samples,
                                np.random.multivariate_normal(p.mu[idx], p.cov[idx], counts[idx])])
    assert samples.shape == (n_sample, dim)
    return samples

def calc_NDB(p, q, n_sample=1000):
    train_samples, test_samples = sample_from(p, n_sample=n_sample), sample_from(q, n_sample=n_sample)
    ndb = NDB(training_data=train_samples)
    results = ndb.evaluate(test_samples)
    return results

def calc_negLogLikelihood(p, q, n_sample=1000):
    train_samples = sample_from(p, n_sample=n_sample)
    return -sum([np.log(MoG_prob_(x, q.pi, q.mu, q.cov)) for x in train_samples]) / n_sample

def calc_FID(p, q, n_sample=1000):
    # \|\mu_p-\mu_q\|_2^2 + Tr(\Sigma_p+\Sigma_q-2\sqrt{\Sigma_p\Sigma_q})
    Mu_p, Mu_q = p.pi.dot(p.mu), q.pi.dot(q.mu)
    # Sigma_p, Sigma_q, calculate sample cov
    train_samples, test_samples = sample_from(p, n_sample=n_sample), sample_from(q, n_sample=n_sample)
    Sigma_p, Sigma_q = np.cov(train_samples.T), np.cov(test_samples.T)
    return np.linalg.norm(Mu_p-Mu_q) ** 2 + np.trace(Sigma_p + Sigma_q - 2*sqrtm(np.dot(Sigma_p, Sigma_q)))

def calc_MMD(p, q, f=lambda x: np.exp(-x**2/2), n_sample=1000):
    def k(x, y):
        return f(np.linalg.norm(x-y))
    train_samples, test_samples = sample_from(p, n_sample=n_sample), sample_from(q, n_sample=n_sample)
    S = 0
    S += sum([k(train_samples[i], train_samples[j]) for i in range(n_sample) for j in range(n_sample)])
    S -= 2 * sum([k(train_samples[i], test_samples[j]) for i in range(n_sample) for j in range(n_sample)])
    S += sum([k(test_samples[i], test_samples[j]) for i in range(n_sample) for j in range(n_sample)])
    return np.sqrt(S) / n_sample

def calc_negELBO(p, q):
    pass

def calc_f(p, q, f, n_sample=1000):
    test_samples = sample_from(q, n_sample=n_sample)
    return sum([f(MoG_prob_(x, p.pi, p.mu, p.cov) / MoG_prob_(x, q.pi, q.mu, q.cov))
                for x in test_samples]) / n_sample

def calc_all_metrics(p, q, n_sample=1000):
    metrics = {}
    tmp = calc_NDB(p, q, n_sample=n_sample)
    metrics['NDB']              = tmp['NDB']
    metrics['JS']               = tmp['JS']
    del tmp
    metrics['negLogLikelihood'] = calc_negLogLikelihood(p, q, n_sample=n_sample)
    metrics['FID']              = calc_FID(p, q, n_sample=n_sample)
    metrics['MMD rbf']          = calc_MMD(p, q, n_sample=n_sample)
    metrics['KL']               = calc_f(p, q, lambda x: x*np.log(x), n_sample=n_sample)
    metrics['revKL']            = calc_f(p, q, lambda x: -np.log(x), n_sample=n_sample)
    metrics['total variation']  = calc_f(p, q, lambda x: np.abs(x-1)/2, n_sample=n_sample)
    metrics['chi square']       = calc_f(p, q, lambda x: (x-1)**2, n_sample=n_sample)
    return metrics

def import_result(file):
    if file[-4:] == '.npz':
        dic = np.load(file)
    else:
        dic = pickle.load(open(file, 'rb'))[0]
    return dic['pi'], dic['mu'], dic['cov']

def main():
    args = parse()
    record = {}

    p = distribution(*import_result(os.path.join('datasets', args.dataset)))  # true

    file = 'results.p'
    q = distribution(*import_result(os.path.join('results', args.result_path, 'full', file)))  # learned
    record[file] = calc_all_metrics(p, q, n_sample=args.n_sample)

    files = os.listdir(os.path.join('results', args.result_path, 'penalized'))
    for file in files:
        if file[:7] == 'results' and file[-2:] == '.p':
            q = distribution(*import_result(os.path.join('results', args.result_path, 'penalized', file)))  # learned
            record[file] = calc_all_metrics(p, q, n_sample=args.n_sample)

    print('\n---------------------results---------------------\n')
    for key in sorted(record.keys()):
        print(key, record[key])
    with open(os.path.join('results', args.result_path, 'metrics.p'), 'wb') as p:
        pickle.dump(record, p)
    all_metrics = list(record['results.p'].keys())
    record_csv = [[key] + [record[key][metric] for metric in all_metrics]
                  for key in sorted(record.keys())]
    np.savetxt(fname=os.path.join('results', args.result_path, 'metrics.csv'),
               X=record_csv,
               fmt='%s',
               delimiter=',',
               header='model,' + ','.join(all_metrics))

if __name__ == '__main__':
    main()