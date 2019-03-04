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

# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = torch.device('cpu')

## Auxiliary classes
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


def MoG_prob(x, pi, mu, cov):
    K, dim = mu.size()
    assert x.size() == (dim,)
    assert pi.size() == (K,)
    assert cov.size() == (K, dim, dim)
    
    priors = torch.softmax(pi, dim=0)
    cov2 = torch.bmm(cov.transpose(1, 2).contiguous(), cov)
    
    prob = 0.0
    for k in range(K):
        # cov2 = torch.matmul(cov[k].t(), cov[k])
        log_prob_k = -dim * 0.5 * math.log(2 * math.pi) - 0.5 * cov2[k].logdet() - 0.5 * cov2[k].inverse().matmul(x - mu[k]).dot(x - mu[k])
        prob += torch.exp(log_prob_k) * priors[k]
    return prob

def MoG_loss(X, z, pi, mu, cov, lam):
    # z: adv_sample
    K, dim = mu.size()
    N = X.size(0)
    assert X.size(1) == dim
    assert pi.size() == (K,)
    assert cov.size() == (K, dim, dim)
    
    loss = lam * MoG_prob(z, pi, mu, cov) if z is not None else 0
    for x in X:
        loss -= torch.log(MoG_prob(x, pi, mu, cov)) / N
    return loss

def GD_solver(train_samples, test_samples, adv_sample=None, lam=2.0, K=5, lr=0.005, max_step=100000, report_step=100, tol=1e-5):
    X_train = torch.FloatTensor(train_samples).to(DEVICE)
    X_test = torch.FloatTensor(test_samples).to(DEVICE)
    z = torch.FloatTensor(adv_sample) if adv_sample is not None else None

    dim = X_train.size(1)
    assert dim == X_test.size(1)

    pi = torch.rand(K, dtype=torch.float).to(DEVICE)
    pi /= pi.sum()
    pi.requires_grad_()
    mu = torch.randn(K, dim, dtype=torch.float).to(DEVICE)
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
    print(MoG_loss(X_train, z, pi, mu, cov, lam=lam))

    optimizer = optim.SGD(params, lr=lr)

    train_p_losses = []     # Primal, no adv term
    train_d_losses = []     # Dual, including adv term; training loss
    test_p_losses = []
    test_d_losses = []
    step_iterator = tqdm(range(max_step))
    for step in step_iterator:
        optimizer.zero_grad()
        loss = MoG_loss(X_train, z, pi, mu, cov, lam=lam)
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
                train_d_loss = MoG_loss(X_train, z, pi, mu, cov, lam=lam)
                train_d_losses.append(train_d_loss.item())
                test_p_loss = MoG_loss(X_test, None, pi, mu, cov, lam=lam)
                test_p_losses.append(test_p_loss.item())
                test_d_loss = MoG_loss(X_test, z, pi, mu, cov, lam=lam)
                test_d_losses.append(test_d_loss.item())

            if len(train_d_losses) > 1 and math.fabs(train_d_losses[-2] - train_d_losses[-1]) < tol:
                step_iterator.close()
                break

    losses = OrderedDict([('train_p_losses', train_p_losses), ('train_d_losses', train_d_losses), ('test_p_losses', test_p_losses), ('test_d_losses', test_d_losses)])

    return pi, mu, cov, losses

if __name__ == '__main__':
    activate_logger('log.txt')
    data_fname = 'data.npz'
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)

    load_data = np.load(data_fname)
    true_pi = load_data['pi']
    true_mu = load_data['mu']
    samples = load_data['samples']
    adv_sample = load_data['adv_sample']

    N = 100
    split_id = -int(N/5)
    samples = samples[:N]
    train_samples = samples[:split_id]
    test_samples = samples[split_id:]

    exps = 3
    # lam_settings = [0.1, 1.0, 10.0]
    lam_settings = [1.0]
    K_settings = [3, 5, 10]
    # K_settings = [5]

    all_settings = [(K, lam) for lam in lam_settings for K in K_settings]

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

    # Adversarial on
    for K, lam in all_settings:
        for e in range(exps):
            print('*** Adversarial on, K = {}, lam = {}, id = {}'.format(K, lam, e + 1))
            output_fname = os.path.join(output_dir, 'result-adv-gd-K={}-lam={}-id={}.npz'.format(K, lam, e + 1))

            pi, mu, cov, losses = GD_solver(train_samples, test_samples, adv_sample, K=K, lam=lam)
            pi = torch.softmax(pi, dim=0).detach().cpu().numpy()
            mu = mu.detach().cpu().numpy()
            cov = torch.bmm(cov.transpose(1, 2), cov).detach().cpu().numpy()

            np.savez(output_fname, pi=pi, mu=mu, cov=cov, **losses)

    deactivate_logger()

