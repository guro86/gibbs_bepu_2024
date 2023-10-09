# -*- coding: utf-8 -*-


import numpy as np
from scipy.stats import norm, uniform 
import emcee
from corner import corner
import pandas as pd
import matplotlib.pyplot as plt


#%%
pi = lambda x1: (
    -1/2*(
        2*x1 + np.sin(6.28*x1)
        )**2
    )

t1 = lambda x: norm(loc=x[0]**3,scale=0.1).logpdf(x[1])
t2 = lambda x: norm(scale=0.1).logpdf(x[1]-x[0]**3)
t3 = lambda x: norm().logpdf((x[1]-x[0]**3)/.1) - np.log(.1)

pi_cond = lambda x1,x2: norm(loc=x1**3,scale=0.1).logpdf(x2)

pi_totti = lambda x: pi(x[0]) + pi_cond(x[0],x[1])

#%%

nwalkers = 10
ndim = 2

nsteps = 1000

initial_state = np.random.randn(nwalkers,ndim)

sampler = emcee.EnsembleSampler(nwalkers, ndim, pi_totti)

res = sampler.run_mcmc(initial_state, nsteps)

#%%

corner(sampler.get_chain(flat=True,discard=500))

#%%

def iterate(n=5000):
    
    x1 = 0
    x2 = 0

    logp_x1 = pi(0)

    proposal = norm(loc=0,scale=.1)

    for i in range(n):
                
        u = uniform.rvs()
    
        x1_cand = x1 + proposal.rvs()
        logp_x1_cand = pi(x1_cand)
        
        if (logp_x1_cand - logp_x1) > np.log(u): 
            x1 = x1_cand
            logp_x1 = logp_x1_cand
        
        x2 = norm(loc=x1**3,scale=0.1).rvs()
        
        yield x1, x2, logp_x1_cand

df = pd.DataFrame.from_records(
    iterate()
    )

#%%

corner(df.iloc[:,:2])

#%%
plt.plot(df.iloc[:,-1])