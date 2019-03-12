import random as rand

import numpy as np
import scipy as sp
from scipy.stats import multivariate_normal as mvn
import os, sys, time, pickle

def log_likelihood(X, pi, mu, sigma):
    ll = 0.0
    for x in X:
        s = 0
        for j in range(len(pi)):
            s += pi[j] * mvn(mu[j], sigma[j]).pdf(x)
        ll += np.log(s)
    return ll

def penalize_term(X, Z, pi, mu, sigma, lam):
    loss = 0
    Z_n = len(Z)
    for z in Z:
        s = 0
        for j in range(len(pi)):
            s += pi[j] * mvn(mu[j], sigma[j]).pdf(z)
        loss += lam*s / Z_n
    return loss
   
def em_gmm(X, pi, mu, sigma, tol=1e-6, max_iter=1000):
    n, p = X.shape
    k = len(pi)

    ll_old = log_likelihood(X, pi, mu, sigma)/n
    losses = [ll_old]
    for i_iter in range(max_iter):
        #print('Iteration %d: log-likelihood is %.6f'%(i_iter, ll_old))

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

        mu = np.zeros((k, p))
        for j in range(k):
            for i in range(n):
                mu[j] += w[j, i] * X[i]
            mu[j] /= w[j, :].sum()

        sigma = np.zeros((k, p, p))
        for j in range(k):
            for i in range(n):
                ys = np.reshape(X[i]- mu[j], (2,1))
                sigma[j] += w[j, i] * np.dot(ys, ys.T)
            sigma[j] /= w[j,:].sum()

        # update complete log likelihoood
        ll_new = log_likelihood(X, pi, mu, sigma)/n
        losses.append(ll_new)

        if np.abs(ll_new - ll_old) < tol:
            print('Terminate! iteration %d: log-likelihood is %.6f'%(i_iter, ll_new))
            break
            
        ll_old = ll_new

    return pi, mu, sigma, losses, i_iter


def em_gmm_penalized(X, Z, pi, mu, sigma, lmda=1, tol=1e-6, max_iter=200):
    n, p = X.shape
    k = len(pi)

    ll= log_likelihood(X, pi, mu, sigma)/n
    loss = ll + penalize_term(X, Z, pi, mu, sigma, lmda)
    p_loss = [ll]
    d_loss = [loss]

    inner_iter_n = []

    for i_iter in range(max_iter):
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
        mu = np.zeros((k, p))
        for j in range(k):
            for i in range(n):
                mu[j] += w[j, i] * X[i]
            mu[j] /= w[j, :].sum()

        sigma = np.zeros((k, p, p))
        for j in range(k):
            for i in range(n):
                ys = np.reshape(X[i]- mu[j], (2,1))
                sigma[j] += w[j, i] * np.dot(ys, ys.T)
            sigma[j] /= w[j,:].sum()

        # M-iter
        old_mu, old_sigma = mu, sigma
        for inner_iter in range(10):
            gamma = np.zeros(k)
            for j in range(k):
                for z in Z:
                    gamma[j] += pi[j]*mvn(old_mu[j], old_sigma[j]).pdf(z)
            gamma *= lmda

            mu = np.zeros((k, p))
            for j in range(k):
                mu[j] -= np.array(gamma[j]*z)
                for i in range(n):
                    mu[j] += w[j, i] * X[i]
                mu[j] /= (w[j, :].sum() - gamma[j])

            sigma = np.zeros((k, p, p))
            for j in range(k):
                v_z_muj = np.reshape(z - old_mu[j],  (2,1))
                sigma[j] -= np.array(gamma[j] * np.dot(v_z_muj, v_z_muj.T))
                for i in range(n):
                    ys = np.reshape(X[i]- old_mu[j], (2,1))
                    sigma[j] += w[j, i] * np.dot(ys, ys.T)
                sigma[j] /= w[j,:].sum()
            
            if np.linalg.norm(old_mu - mu) + np.linalg.norm(old_sigma - sigma)  < 1e-16: #convergence criterion
                old_mu, old_sigma = mu, sigma
                break

            old_mu, old_sigma = mu, sigma
        
        #M-iter ends
        inner_iter_n.append(inner_iter)

        # update complete log likelihoood
        ll = log_likelihood(X, pi, mu, sigma)/n
        loss_ = ll + penalize_term(X, Z, pi, mu, sigma, lmda)
        p_loss.append(ll)
        d_loss.append(loss_)

        if np.abs(loss_ - loss) < tol:
            print('Terminate! iteration %d: log-likelihood is %.6f'%(i_iter, loss_))
            break
            
        loss = loss_

    return pi, mu, sigma, p_loss, d_loss, inner_iter_n


if __name__ == '__main__':
    
    output_dir = 'results_multi_adv_EM'
    os.makedirs(output_dir,exist_ok=True)

    data_fname = 'data_multi_adv.npz'
    load_data = np.load(data_fname)
    N = 500
    true_pi = load_data['pi']
    true_mu = load_data['mu']
    X = load_data['samples'][:N]
    Z = load_data['adv_sample']
    
    exps = 5
    lam_settings = [0.01, 0.1, 1.0, 10.0]
    K_settings = [3, 5, 10]
    all_settings = [(K, lam) for lam in lam_settings for K in K_settings]

    for K, lam in all_settings:
        em_results = []
        em_p_results = []
        
        for e in range(exps):
            try:
                # initial guesses for parameters
                pis = np.ones(K)
                pis /= pis.sum()
                mus = np.random.random((K,2))
                sigmas = np.array([np.eye(2)] * K)

                start_t = time.time()
                pi, mu, conv, losses, iter_n = em_gmm(X, pis, mus, sigmas)
                em_results.append({"init_guess":[pis, mus, sigmas],
                                    "pi":pi, "mu":mu, "conv": conv,
                                    "loss":losses, "iters":iter_n, 
                                    "time":time.time()-start_t})
                
                start_t = time.time()            
                pi, mu, conv, p_loss, d_loss, inner_iter = em_gmm_penalized(X, Z, pis, mus, sigmas)
                em_p_results.append({"init_guess":[pis, mus, sigmas],
                                    "pi":pi, "mu":mu, "conv": conv, 
                                    "p_loss":p_loss, "d_loss":d_loss, 
                                    "iters":inner_iter,
                                    "time":time.time()-start_t})
            except:
                print("Error; singular matrix")

        with open(output_dir+'/EM-K={}-lam={}-N={}.p'.format(K, lam, N), 'wb') as p:
            pickle.dump(em_results, p)

        with open(output_dir+'/Penalized-K={}-lam={}-N={}.p'.format(K, lam, N), 'wb') as p:
            pickle.dump(em_p_results, p)