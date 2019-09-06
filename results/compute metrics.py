import numpy as np
import os

def read(file):
    data = np.loadtxt(file, delimiter=',', dtype=str)
    header = open(file).readline()
    return header, data

def main(folder, selected):
    # read data
    data = {}
    header = []
    nrow = 2
    for i in range(10):
        header, data[i] = read(os.path.join(folder, 'metrics'+str(i)+'.csv'))
        nrow = len(data[i])

    # set headers
    headers = header.split(',')
    for j in range(len(headers)):
        if '#' in headers[j]:
            headers[j] = headers[j][2:]
        if '\n' in headers[j]:
            headers[j] = headers[j][:-1]
        if headers[j] == 'MMD rbf':
            headers[j] = 'MMD'
        if headers[j] == 'negLogLikelihood':
            headers[j] = '$-\log\ell$'

    # set results
    results = {i:{} for i in range(nrow)}
    for j in range(1, len(headers)):
        for i in range(nrow):
            array = [float(d[i][j]) for d in data.values()]
            results[i][headers[j]] = (np.mean(array), np.std(array),)

    # output as latex format
    backslash = '\\'
    trun = lambda x: np.around(x, decimals=3)
    X = [[' & '.join(['Model']+selected) + backslash + backslash + ' ' + backslash + 'hline']]
    for i in range(nrow):
        if i < nrow - 1:
            X.append([' & '.join(
                [data[0][i][0]] + [str(trun(results[i][h][0])) + ' $'+backslash+'pm$ ' + str(trun(results[i][h][1]))
                                   for h in selected]) + backslash + backslash])
        else:
            X.append([' & '.join(
                [data[0][i][0]] + [str(trun(results[i][h][0])) + ' $' + backslash + 'pm$ ' + str(trun(results[i][h][1]))
                                   for h in selected]) + backslash + backslash + ' ' + backslash + 'hline'])

    np.savetxt(fname=os.path.join(folder + '.txt'),
               X=X,
               fmt='%s')

if __name__ == '__main__':
    selected = ['NDB', '$-\log\ell$', 'KL', 'JS', 'MMD']
    main('metricsagain', selected)