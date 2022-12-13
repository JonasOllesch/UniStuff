import numpy as np
np.random.seed(1337)
bins = 100
def random_poisson(lam,n):
    return np.random.poisson(lam=lam,size=n)


def random_normal(lam,n):
    return np.random.normal(loc=lam, scale=lam, size=n)

lam = 1
rnd_poisson_num = random_poisson(lam,10)
rnd_normal_num = random_normal(lam,10)

rnd_poisson_num = np.rint(rnd_poisson_num)
rnd_normal_num = np.rint(rnd_normal_num)

min_bin=lam-5*np.sqrt(lam)
max_bin=lam+5*np.sqrt(lam)

binsize = (max_bin-min_bin)/bins
poi_binned = np.zeros(100)
nor_binned = np.zeros(100)

#poi_binned[i] = np.sum(rnd_normal_num[>])

#print(rnd_poisson_num)
#print(rnd_normal_num)