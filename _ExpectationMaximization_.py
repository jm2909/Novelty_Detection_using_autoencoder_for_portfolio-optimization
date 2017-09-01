
import scipy
import  numpy as np
import pandas as pd
class SingleseriesGaussianMixture():
    def __init__(self,series,n_components):
        self.series = series
        self.n_components  =n_components
    def __normaldensity__(self,value,mean,sigma):
        return scipy.stats.norm.pdf(value,mean,sigma)
    def __initializer(self):
        mu = np.zeros(self.n_components)
        sd = np.ones(self.n_components)
        sum = np.zeros(self.n_components)
        count = np.zeros(self.n_components)
        sumsq = np.zeros(self.n_components)
        return mu,sd,sum,count,sumsq

    def __Expectation__(self,value,mu_array,sigma_array):
        ncompo = mu_array.shape[0]
        prob = np.array([self.__normaldensity__(value,mu_array[j],sigma_array[j]) for j in range(0,ncompo)])
        label = np.argmax(prob)
        return label

    def __maximization__(self,value,mu_array,sigma_array,sum_array,count,sumsq,labellist):
        cluster = self.__Expectation__(value,mu_array,sigma_array)
        labellist.append(cluster)
        sum_array[cluster] = sum_array[cluster] + value
        count[cluster] = count[cluster] + 1
        mu_array[cluster] = sum_array[cluster] / float(count[cluster])
        sumsq[cluster] = sumsq[cluster]+(value**2)
        if count[cluster] == 1:
            sigma_array[cluster]  = 1
        else:
            sigma_array[cluster] = np.sqrt((sumsq[cluster]/float(count[cluster]) - (mu_array[cluster]**2)))
        return mu_array,sigma_array,sum_array,count,sumsq,labellist

    def __EM__(self):
        lists = []
        mu, sd, sum, count, sumsq = self.__initializer()
        for xx in self.series:
            mu_array, sigma_array, sum_array, count, sumsq, labellist = self.__maximization__(xx, mu_array=mu,
                                                                                         sigma_array=sd, sum_array=sum,
                                                                                         count=count, sumsq=sumsq,
                                                                                         labellist=lists)
        return mu_array, sigma_array, sum_array, count, sumsq,pd.DataFrame(np.array(labellist), columns=['Labels'])
