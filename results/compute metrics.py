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
    for i in range(10):
        header, data[i] = read(os.path.join(folder, 'metrics'+str(i)+'.csv'))

    # set headers
    headers = header.split(',')
    for j in range(len(headers)):
        if '#' in headers[j]:
            headers[j] = headers[j][2:]
        if '\n' in headers[j]:
            headers[j] = headers[j][:-1]

    # set results
    resultsEM, resultspenEM = {}, {}
    for j in range(1, len(headers)):
        array = [float(d[0][j]) for d in data.values()]
        resultsEM[headers[j]] = (np.mean(array), np.std(array),)
        array = [float(d[1][j]) for d in data.values()]
        resultspenEM[headers[j]] = (np.mean(array), np.std(array),)

    # writa mean to csv
    savecsv = False
    if savecsv:
        X = [['EM'] + [resultsEM[h][0] for h in headers[1:]],
              ['penEM'] + [resultspenEM[h][0] for h in headers[1:]],
              ['EM std'] + [resultsEM[h][1] for h in headers[1:]],
              ['penEM std'] + [resultspenEM[h][1] for h in headers[1:]]]
        np.savetxt(fname=os.path.join(folder+'.csv'),
                   X=X,
                   fmt='%s',
                   delimiter=',',
                   header=header)

    # output as latex format

    backslash = '\\'
    trun = lambda x: np.around(x, decimals=3)
    X = [[' & '.join(['model']+selected) + backslash + backslash + ' ' + backslash + 'hline']]
    X.append([' & '.join(['EM']   +[str(trun(resultsEM[h][0]))    + ' $'+backslash+'pm$ ' + str(trun(resultsEM[h][1]))
                                    for h in selected]) + backslash + backslash])
    X.append([' & '.join(['penEM']+[str(trun(resultspenEM[h][0])) + ' $'+backslash+'pm$ ' + str(trun(resultspenEM[h][1]))
                                    for h in selected]) + backslash + backslash + ' ' + backslash + 'hline'])
    np.savetxt(fname=os.path.join(folder + '.txt'),
               X=X,
               fmt='%s')

if __name__ == '__main__':
    selected = ['NDB', 'negLogLikelihood', 'KL', 'JS', 'MMD rbf', 'total variation', 'chi square']
    main('1dmetrics', selected)
    main('2dmetrics', selected)