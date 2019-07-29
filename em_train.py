import random as rand
import numpy as np
import scipy as sp
from scipy.stats import multivariate_normal as mvn
from scipy import linalg
from scipy.special import logsumexp
import os, sys, time, pickle
from tqdm import tqdm
import argparse


class Logger(object):
    def __init__(self, log_fname):
        self.terminal = sys.stdout
        self.log = open(log_fname, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        pass

def activate_logger(log_fname):
    logger = Logger(log_fname)
    sys.stdout = logger

def deactivate_logger():
    sys.stdout.log.close()
    sys.stdout = sys.stdout.terminal

def penalize_term(X, Z, pi, mu, sigma, lam):
    loss = 0
    Z_n = len(Z)
    for z in Z:
        s = 0
        for j in range(len(pi)):
            s += pi[j] * mvn(mu[j], sigma[j]).pdf(z)
        loss += lam*s / Z_n
    return loss


#precision is the inverse of covariance matrix
def compute_precision_cholesky(covariances):
    n_components, n_features, _ = covariances.shape
    precisions_chol = np.empty((n_components, n_features, n_features))
    for k, covariance in enumerate(covariances):
        try:
            cov_chol = linalg.cholesky(covariance, lower=True)
        except linalg.LinAlgError:
            raise ValueError(estimate_precision_error_message)
        precisions_chol[k] = linalg.solve_triangular(cov_chol,
                                                     np.eye(n_features),
                                                     lower=True).T
    return precisions_chol


def compute_log_det_cholesky(matrix_chol, n_features):
    n_components, _, _ = matrix_chol.shape
    log_det_chol = (np.sum(np.log(
        matrix_chol.reshape(n_components, -1)[:, ::n_features + 1]), 1))
    return log_det_chol


def log_prob_cholesky(X, means, precisions_chol):

    n_samples, n_features = X.shape
    n_components, _ = means.shape
    # det(precision_chol) is half of det(precision)
    log_det = compute_log_det_cholesky(precisions_chol, n_features)

    log_prob = np.empty((n_samples, n_components))
    for k, (mu, prec_chol) in enumerate(zip(means, precisions_chol)):
        y = np.dot(X, prec_chol) - np.dot(mu, prec_chol)
        log_prob[:, k] = np.sum(np.square(y), axis=1)

    return -.5 * (n_features * np.log(2 * np.pi) + log_prob) + log_det

def mean_log_likelihood(X, pi, means, precisions_chol):
    weighted_log_prob = log_prob_cholesky(X, means, precisions_chol) + np.log(pi)
    return logsumexp(weighted_log_prob, axis=1).mean()

   
def em_gmm(X, pi, mu, sigma, tol=1e-6, max_iter=1000):
    n, d = X.shape
    k = len(pi)

    step_iterator = tqdm(range(max_iter))
    
    precisions_cholesky = compute_precision_cholesky(sigma)
    ll_old = mean_log_likelihood(X, pi, mu, precisions_cholesky)
    losses = [ll_old]
    
    for i_iter in step_iterator:
        #print('EM Iteration %d: log-likelihood is %.6f'%(i_iter, ll_old))

        # E-step
        w = np.zeros((k, n))
        for j in range(len(mu)):
            for i in range(n):
                w[j, i] = pi[j] * mvn(mu[j], sigma[j]).pdf(X[i])
        w /= w.sum(0)
        
        
        # M-step
        pi = np.zeros(k)
        for j in range(len(mu)):
            for i in range(n):
                pi[j] += w[j, i]
        pi /= n

        mu = np.zeros((k, d))
        for j in range(k):
            for i in range(n):
                mu[j] += w[j, i] * X[i]
            mu[j] /= w[j, :].sum()

        sigma = np.zeros((k, d, d))
        for j in range(k):
            for i in range(n):
                ys = np.reshape(X[i]- mu[j], (d,1))
                sigma[j] += w[j, i] * np.dot(ys, ys.T) 
            sigma[j] /= w[j,:].sum()
            sigma[j] += np.eye(d) * 1e-6 #for possitive-ness
        precisions_cholesky = compute_precision_cholesky(sigma)
        
        # update complete log likelihoood
        ll_new = mean_log_likelihood(X, pi, mu, precisions_cholesky)
        losses.append(ll_new)

        if np.abs(ll_new - ll_old) < tol:
            print('Terminate! iteration %d: log-likelihood is %.6f'%(i_iter, ll_new))
            step_iterator.close()
            break
            
        ll_old = ll_new

    return pi, mu, sigma, losses, i_iter


def em_gmm_penalized(X, Z, pi, mu, sigma, lmda=1, tol=1e-6, max_iter=1000):
    n, d = X.shape
    k = len(pi)

    precisions_cholesky = compute_precision_cholesky(sigma)
    ll = mean_log_likelihood(X, pi, mu, precisions_cholesky)
    
    loss = ll + penalize_term(X, Z, pi, mu, sigma, lmda)
    p_loss = [ll]
    d_loss = [loss]

    inner_iter_n = []
    step_iterator = tqdm(range(max_iter))
    for i_iter in step_iterator:
        #print('Iteration %d: log-likelihood is %.6f'%(i_iter, loss))

        # E-step
        w = np.zeros((k, n)) #priors
        for j in range(len(mu)):
            for i in range(n):
                w[j, i] = pi[j] * mvn(mu[j], sigma[j]).pdf(X[i])
        w /= w.sum(0)

        # M-step
        pi = np.zeros(k)
        for j in range(len(mu)):
            for i in range(n):
                pi[j] += w[j, i]
        pi /= n
        
        # M-iter initialization
        mu = np.zeros((k, d))
        for j in range(k):
            for i in range(n):
                mu[j] += w[j, i] * X[i]
            mu[j] /= w[j, :].sum()

        sigma = np.zeros((k, d, d))
        for j in range(k):
            for i in range(n):
                ys = np.reshape(X[i]- mu[j], (d,1))
                sigma[j] += w[j, i] * np.dot(ys, ys.T)
            sigma[j] /= w[j,:].sum()

        # M-iter
        old_mu, old_sigma = mu, sigma

        for inner_iter in range(100):
            gamma = np.zeros((len(Z),k))
            
            for iz, z in enumerate(Z):
                for j in range(k):
                    gamma[iz, j] += pi[j]*mvn(old_mu[j], old_sigma[j]).pdf(z)

            mu = np.zeros((k, d))
            for j in range(k):
                mu[j] -= lmda * np.dot(gamma[:,j],Z)
                for i in range(n):
                    mu[j] += w[j, i] * X[i]
                mu[j] /= w[j, :].sum() - lmda * gamma[:,j].sum()

            sigma = np.zeros((k, d, d))
            for j in range(k):
                for iz, z in enumerate(Z):
                    z_diff = np.reshape(z - mu[j], (d,1))
                    sigma[j] -= gamma[iz,j] * np.dot(z_diff, z_diff.T)
                sigma[j] *= lmda 
                
                for i in range(n):
                    ys = np.reshape(X[i]- old_mu[j], (d, 1))
                    sigma[j] += w[j, i] * np.dot(ys, ys.T)
                sigma[j] /= w[j,:].sum() - lmda * gamma[:,j].sum()
            
            if np.linalg.norm(old_mu - mu) + np.linalg.norm(old_sigma - sigma)  < 1e-10: #tol_inner
                old_mu, old_sigma = mu, sigma
                break

            old_mu, old_sigma = mu, sigma
        
        #M-iter ends
        inner_iter_n.append(inner_iter)
        precisions_cholesky = compute_precision_cholesky(sigma)

        # update complete log likelihoood
        ll = mean_log_likelihood(X, pi, mu, precisions_cholesky)
        loss_ = ll + penalize_term(X, Z, pi, mu, sigma, lmda)
        p_loss.append(ll)
        d_loss.append(loss_)

        if np.abs(loss_ - loss) < tol:
            print('Terminate! iteration %d: log-likelihood is %.6f'%(i_iter, loss_))
            step_iterator.close()
            break
            
        loss = loss_

    return pi, mu, sigma, p_loss, d_loss, inner_iter_n


def main(): # "standard" or "penalized"

    parser = argparse.ArgumentParser(description='em_train parser')
    parser.add_argument('--algo', type=str, default='penalized', help='regular or penalized')
    parser.add_argument('--output_dir', type=str, default='results', help='output directory')
    parser.add_argument('--dataset_name', type=str, default='MNIST', help='dataset name')
    parser.add_argument('--dataset_file', type=str, default='mnist_1797_afterPCA_adv.npz', help='dataset file name')

    parser.add_argument('--num_experiments', type=int, default=1, help='# of times each experiment setting will be performed')
    parser.add_argument('--k', type=int, default=10, help='number of Gaussians')
    parser.add_argument('--lam', type=float, default=10.0, help='lambda value')

    args = parser.parse_args()

    activate_logger('log-EM-%s.txt'%args.algo)

    data_fname = os.path.join('datasets', args.dataset_name, args.dataset_file)
    load_data = np.load(data_fname)

    output_path = os.path.join(args.output_dir, args.dataset_name, args.algo)
    os.makedirs(output_path, exist_ok=True)
    with open(os.join(output_path,'args.txt'), 'w') as f:
        f.write('\n'.join(sys.argv[1:]))

    X = load_data['samples']
    Z = load_data['adv_sample']
    data_range = 1.0
    d = X.shape[1]
    
    K = args.k
    lam = args.lam
    
    for e in range(args.num_experiments):
        results = []
        print("k:", K,"lam:", lam)
        
        # initial guesses for parameters
        pis = np.ones(K)
        pis /= pis.sum()
        mus = np.random.random((K,d)) * data_range 
        sigmas = np.array([np.eye(d)] * K)
        
        if args.algo == "regular":
            start_t = time.time()
            pi, mu, conv, losses, iter_n = em_gmm(X, pis, mus, sigmas)
            results.append({"init_guess":[pis, mus, sigmas],
                                "pi":pi, "mu":mu, "conv":conv,
                                "loss":losses, "iters":iter_n, 
                                "time":time.time()-start_t})
        
        elif args.algo == "penalized":
            start_t = time.time()
            pi, mu, conv, p_loss, d_loss, inner_iter = em_gmm_penalized(X, Z, pis, mus, sigmas, lam)
            results.append({"init_guess":[pis, mus, sigmas],
                                "pi":pi, "mu":mu, "conv":conv, 
                                "p_loss":p_loss, "d_loss":d_loss, 
                                "iters":inner_iter,
                                "time":time.time()-start_t})


        with open(os.path.join(output_path, 'K={}-lam={}-N={}.p'.format(K, lam, N)), 'wb') as p:
            pickle.dump(results, p)

    deactivate_logger()

if __name__ == '__main__':
    main()
    
