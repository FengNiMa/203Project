import os, sys, time, pickle
import argparse, json
from algorithm import *

def parse():

    parser = argparse.ArgumentParser(description='em_train parser')

    parser.add_argument('--algo', type=str, default='penalized', help='standard or penalized')
    parser.add_argument('--output_dir', type=str, default='results', help='output directory')
    parser.add_argument('--dataset_file', type=str, default='multi-adv-0/data_multi_adv.npz', help='path to dataset file')
    parser.add_argument('--load_path', type=str, default='', help='Path to saved result file')

    parser.add_argument('--num_experiments', type=int, default=1, help='# of times each experiment setting will be performed')
    parser.add_argument('--k', type=int, default=5, help='number of Gaussians')
    parser.add_argument('--lam', type=float, default=10.0, help='lambda value')

    parser.add_argument('--index', type=int, default=-1, help='index of online training')

    return parser.parse_args()

class Logger(object):
    def __init__(self, log_fname):
        self.terminal = sys.stdout
        self.log = open(log_fname, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        pass

    def activate(self):
        sys.stdout = self

    def deactivate(self):
        sys.stdout.log.close()
        sys.stdout = sys.stdout.terminal

def init_guess(K, d):
    pis = np.ones(K)
    pis /= pis.sum()
    mus = np.random.random((K,d))
    sigmas = np.array([np.eye(d)] * K)
    return pis, mus, sigmas

def import_result(fname):
    with open(fname, "rb") as p:
        res = pickle.load(p)
    pi = [r["pi"] for r in res]
    mu = [r["mu"] for r in res]
    sigma = [r["cov"] for r in res]
    return pi[0], mu[0], sigma[0]


def main():

    args = parse()

    # set up outout
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir,'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    logger = Logger(os.path.join(args.output_dir,'log.txt'))
    logger.activate()

    # load data
    data_fname = os.path.join('datasets', args.dataset_file)
    load_data = np.load(data_fname)
    X = load_data['samples']
    Z = load_data['adv_sample']
    lam = args.lam
    d = X.shape[1]
    
    # if continue learning
    if len(args.load_path) > 0:
        prev_pis, prev_mus, prev_sigmas = import_result(args.load_path)
        K = len(prev_pis)
    else:
        K = args.k


    for e in range(args.num_experiments):
        results = []
        print("k:", K,"lam:", lam)
        
        # initial guesses for parameters
        if len(args.load_path) > 0:
            pis, mus, sigmas = prev_pis, prev_mus, prev_sigmas
        else:
            pis, mus, sigmas = init_guess(K, d)

        if args.algo == "standard":
            start_t = time.time()
            pi, mu, cov, losses, iter_n = em_gmm(X, pis, mus, sigmas)
            results.append({"init":[pis, mus, sigmas],
                            "pi":pi, "mu":mu, "cov":cov,
                            "loss":losses, "iters":iter_n, 
                            "time":time.time()-start_t})
        
        elif args.algo == "penalized":
            start_t = time.time()
            pi, mu, cov, p_loss, d_loss, inner_iter = em_gmm_penalized(X, Z, pis, mus, sigmas, lam)
            results.append({"init":[pis, mus, sigmas],
                            "pi":pi, "mu":mu, "cov":cov, 
                            "p_loss":p_loss, "d_loss":d_loss, 
                            "iters":inner_iter,
                            "time":time.time()-start_t})
        else:
            print("invalid algorithm specification. Only support 'standard' or 'penalized'. ")

        if args.index == -1:
            filename = 'results.p'
        else:
            filename = 'results-' + str(args.index) + '.p'
        with open(os.path.join(args.output_dir, filename), 'wb') as p:
            pickle.dump(results, p)

    logger.deactivate()

if __name__ == '__main__':
    main()
