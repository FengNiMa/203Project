import math
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal

# from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm
from collections import OrderedDict
import os, sys
import json

#DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = torch.device('cpu')


def MoG_prob(x, pi, mu, cov, min_eig=1e-3):
    K, dim = mu.size()
    assert x.size() == (dim,)
    assert pi.size() == (K,)
    assert cov.size() == (K, dim, dim)
    
    priors = torch.softmax(pi, dim=0)
    # cov2 = torch.bmm(cov.transpose(1, 2).contiguous(), cov)
    
    prob = 0.0
    for k in range(K):
        cov2 = torch.matmul(cov[k].t(), cov[k]) + torch.eye(dim) * min_eig
        log_prob_k = -dim * 0.5 * math.log(2 * math.pi) - 0.5 * cov2.logdet() - 0.5 * cov2.inverse().matmul(x - mu[k]).dot(x - mu[k])
        prob += torch.exp(log_prob_k) * priors[k]
    return prob

def MoG_loss(X, Z, pi, mu, cov, lam):
    # Z: adv_samples, can be None
    K, dim = mu.size()
    N = X.size(0)
    assert X.size(1) == dim
    assert pi.size() == (K,)
    assert cov.size() == (K, dim, dim)
    
    loss = 0
    if Z is not None:
        N_z = Z.size(0)
        for z in Z:
            loss += lam * MoG_prob(z, pi, mu, cov) / N_z
    for x in X:
        loss -= torch.log(MoG_prob(x, pi, mu, cov)) / N
    return loss

def GD_solver(train_samples, adv_samples=None, data_range=6.0, lam=1.0, K=5, lr=0.002, max_step=100000, report_step=100, tol=1e-5, patience=10):
    X_train = torch.FloatTensor(train_samples).to(DEVICE)
    #X_test = torch.FloatTensor(test_samples).to(DEVICE)
    Z = torch.FloatTensor(adv_samples) if adv_samples is not None else None

    dim = X_train.size(1)
    #assert dim == X_test.size(1)
    assert Z is None or dim == Z.size(1)

    pi = torch.rand(K, dtype=torch.float).to(DEVICE)
    pi /= pi.sum()
    pi.requires_grad_()
    mu = torch.rand(K, dim, dtype=torch.float).to(DEVICE) * data_range
    mu.requires_grad_()
    cov = torch.eye(dim, dtype=torch.float).repeat(K, 1, 1).to(DEVICE)
    cov.requires_grad_()

    params = [pi, mu, cov]
    named_params = OrderedDict([('pi', pi), ('mu', mu), ('cov', cov)])

    print('*** Init ***')
    for n, p in named_params.items():
        print(n)
        print(p)
        def _hook(grad, p=p, n=n):
            if torch.isnan(grad).sum() > 0:
                print('Error in grad:', grad)
                print('Shape:', grad.size())
                print('Parameter name:', n)
                print('p.requires_grad:', p.requires_grad)
                raise ValueError
        p.register_hook(_hook)
    print('loss:')
    print(MoG_loss(X_train, Z, pi, mu, cov, lam=lam))

    # optimizer = optim.SGD(params, lr=lr)
    optimizer = optim.Adam(params, lr=lr)

    train_p_losses = []     # Primal, no adv term
    train_d_losses = []     # Dual, including adv term; training loss
    #test_p_losses = []
    #test_d_losses = []
    step_iterator = tqdm(range(max_step))

    no_improve = 0
    best_train_loss = 1e10
    for step in step_iterator:
        optimizer.zero_grad()
        loss = MoG_loss(X_train, Z, pi, mu, cov, lam=lam)
        loss.backward()
        optimizer.step()
        if (step + 1) % report_step == 0:
            print('Step {}'.format(step + 1))
            for n, p in named_params.items():
                print(n)
                print(p.data)
            print('training loss:')
            print(loss.item())
            print()

            with torch.no_grad():
                train_p_loss = MoG_loss(X_train, None, pi, mu, cov, lam=lam)
                train_p_losses.append(train_p_loss.item())
                train_d_loss = MoG_loss(X_train, Z, pi, mu, cov, lam=lam)
                train_d_losses.append(train_d_loss.item())
                #test_p_loss = MoG_loss(X_test, None, pi, mu, cov, lam=lam)
                #test_p_losses.append(test_p_loss.item())
                #test_d_loss = MoG_loss(X_test, Z, pi, mu, cov, lam=lam)
                #test_d_losses.append(test_d_loss.item())

            if loss.item() < best_train_loss - tol:
                no_improve = 0
                best_train_loss = loss.item()
            else:
                no_improve += 1
            if no_improve > patience:
                step_iterator.close()
                break

    losses = OrderedDict([('train_p_losses', train_p_losses), ('train_d_losses', train_d_losses)]) #, ('test_p_losses', test_p_losses), ('test_d_losses', test_d_losses)])

    return pi, mu, cov, losses

if __name__ == '__main__':
    activate_logger('log.txt')

    output_dir = 'results'
    dataset_name = 'multi-adv-2'
    os.makedirs(os.path.join(output_dir, dataset_name, 'GD'), exist_ok=True)

    data_fname = os.path.join('datasets', dataset_name, 'data_multi_adv_1000.npz')

    load_data = np.load(data_fname)
    true_pi = load_data['pi']
    true_mu = load_data['mu']
    samples = load_data['samples']
    adv_samples = load_data['adv_sample']
    dim = samples.shape[1]

    N = 1000
    #split_id = -int(N/5)
    train_samples = samples[:N]
    #train_samples = samples[:split_id]
    #test_samples = samples[split_id:]

    exps = 1
    lam_settings = [10.0, 100.0, 1000.0]
    # lam_settings = [1.0]
    K_settings = [3, 5, 10]
    # K_settings = [10]
    min_eig = 1e-3
    
    all_settings = [(K, lam) for lam in lam_settings for K in K_settings]
    '''
    # Adversarial off
    for K in K_settings:
        for e in range(exps):
            print('*** Adversarial off, K = {}, id = {}'.format(K, e + 1))
            output_fname = os.path.join(output_dir, 'result-nonadv-gd-K={}-id={}.npz'.format(K, e + 1))

            pi, mu, cov, losses = GD_solver(train_samples, test_samples, None, K=K)
            pi = torch.softmax(pi, dim=0).detach().cpu().numpy()
            mu = mu.detach().cpu().numpy()
            cov = torch.bmm(cov.transpose(1, 2), cov).detach().cpu().numpy()

            np.savez(output_fname, pi=pi, mu=mu, cov=cov, **losses)
    '''
    # Adversarial on
    for K, lam in all_settings:
        p_losses = []
        d_losses = []
        for e in range(exps):
            print('*** Adversarial on, K = {}, lam = {}, id = {}'.format(K, lam, e + 1))
            output_fname = os.path.join(output_dir, dataset_name, 'GD', 'result-adv-adam-K={}-lam={}-N={}-id={}.npz'.format(K, lam, N, e + 1))

            pi, mu, cov, losses = GD_solver(train_samples, adv_samples, K=K, lam=lam, lr=1e-3)
            p_losses.append(losses["train_p_losses"])
            d_losses.append(losses["train_d_losses"])
            pi = torch.softmax(pi, dim=0).detach().cpu().numpy()
            mu = mu.detach().cpu().numpy()
            cov = torch.bmm(cov.transpose(1, 2), cov).detach().cpu().numpy()
            cov += np.tile(np.eye(dim), (K, 1, 1)) * min_eig  # Used in MoG_prob

            np.savez(output_fname, pi=pi, mu=mu, cov=cov, **losses)
        
        with open(os.path.join(output_dir, dataset_name, 'GD', 'losses-adam-K={}-lam={}-N={}.json'.format(K, lam, N)), 'w') as outfile:
            json.dump({"p_loss":p_losses, "d_loss":d_losses}, outfile)
    
    deactivate_logger()

