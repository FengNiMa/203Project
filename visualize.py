import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import math, os, argparse, pickle


def parse():
    parser = argparse.ArgumentParser(description='em result visualizer')
    parser.add_argument('file_path', type=str, help='file to visualize.\
                                                     (if results: directory name, \
                                                      if dataset: path to .npz file')
    parser.add_argument('--save_path', type=str, default='plot.jpg', help='path to save image')
    return parser.parse_args()

def import_result(file):
    dic = pickle.load(open(file, 'rb'))[0]
    return dic['pi'], dic['mu'], dic['cov']

def import_data(file):
    data_fname = os.path.join('datasets', file)
    load_data = np.load(data_fname)
    return load_data['samples'], load_data['adv_sample']


def MoG_prob_(X,phi,mu,cov):
    phi = np.array(phi)
    mu = np.array(mu)
    cov = np.array(cov)

    probs = []
    for i,mean in enumerate(mu):
        a = multivariate_normal.pdf(X, mean, cov[i])
        probs.append(a)

    p = np.dot(np.array(probs).T, phi)
    return p

def MoG_plot(pi, mu, cov, save_path):
    plt.figure(figsize=(5, 5))

    x1 = x2 = np.linspace(-2.0, 8.0, 101)
    p_lists = []
    X = []
    for _x1 in x1:
        for _x2 in x2:
            X.append([_x1,_x2])
    P = MoG_prob_(X, pi, mu, cov).reshape((len(x2),len(x1)))

    plt.imshow(P, origin='lower', interpolation='bilinear')
    plt.xticks(np.linspace(0, 100, 6), np.linspace(-2, 8, 6))
    plt.yticks(np.linspace(0, 100, 6), np.linspace(-2, 8, 6))
    plt.savefig(save_path)

def data_plot(samples, adv_sample, save_path):
    plt.figure(figsize=(5, 5))

    x1 = [x[0] for x in samples]
    x2 = [x[1] for x in samples]
    plt.scatter(x1, x2, s=5, marker='o')
    
    x1 = [x[0] for x in adv_sample]
    x2 = [x[1] for x in adv_sample]
    plt.scatter(x1, x2, s=10, marker='X')
    
    plt.xlim(-2, 8)
    plt.ylim(-2, 8)
    plt.savefig(save_path)

def main():
    args = parse()
    save_path = os.path.join("figures", args.save_path)
    directory = os.path.dirname(save_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    file = args.file_path
    ftype = file.split(".")[-1]
    # dataset file
    if ftype == "npz":
        x, z = import_data(file)
        data_plot(x, z, save_path)
    # everything below is result
    elif ftype == "p":
        pi, mu, cov = import_result(os.path.join('results', file))
        MoG_plot(pi,mu,cov,save_path)
    else:
        file = os.path.join('results', file, 'results.p')
        if os.path.exists(file):
            pi, mu, cov = import_result(file)
            MoG_plot(pi,mu,cov,save_path)   
        elif os.path.exists(os.path.join('results', file, 'results.p')):
            pi, mu, cov = import_result(os.path.join( file, 'results.p'))
            MoG_plot(pi,mu,cov,save_path)   
        else:
            print("Please enter valid file name (end with .p or .npz or is a directory)")

if __name__ == '__main__':
    main()





