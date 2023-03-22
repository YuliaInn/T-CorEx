from tcorex import corex
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
import os
import time

def main():
    X = pd.read_csv("/Users/yuliai/PycharmProjects/LinearCorex/data/d1s4090_HVGs/d1s4090_norm2000.csv", header = 0, index_col = 0)
    def optim_n(X, nmin, nmax):
        nll = []
        for n in range(nmin, nmax):
            out = corex.Corex(2000, n_hidden=n, verbose = 2).fit(X)
            cov_mat = pd.DataFrame(data=out.get_covariance(), index=X.columns ,columns=X.columns)
            mean = np.mean(X, axis=0)
            std = np.sqrt(np.sum((X - mean)**2, axis=0) / 2000).clip(1e-10)
            theta = (mean, std)
            Xz = ((X - theta[0]) / theta[1])
            nll.append(-multivariate_normal.logpdf(Xz,cov=cov_mat).mean())
        return(nll)

    nll = optim_n(X, 1, 5)
    fig = plt.figure()
    plt.plot(nll)
    fig.savefig('nll_plot.png')
    min_value = min(nll)
    min_index = nll.index(min_value) + 1

    out = corex.Corex(2000, n_hidden=min_index, verbose = 2).fit(X)
    cov_mat = pd.DataFrame(data=out.get_covariance(), index=X.columns ,columns=X.columns)
    mis = pd.DataFrame(data=out.mis(), index=list(range(min_index)),columns=X.columns)
    mis.to_csv("mis.csv")
    cov_mat.to_csv("cov_mat.csv")

if __name__ == '__main__':
    folder_name = 'T-CorEx_' + time.strftime("%Y_%m_%d_%H_%M_%S")
    os.mkdir(folder_name)
    main()