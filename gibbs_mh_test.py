# -*- coding: utf-8 -*-


import numpy as np
from scipy.stats import norm, uniform 
import emcee
from corner import corner
import pandas as pd
import matplotlib.pyplot as plt


#%% PDF definitions

#logp of x1
pi_x1 = lambda x1: (
    -1/2*(
        2*x1 + np.sin(6.28*x1)
        )**2
    )

#logp of x2 given x1
pi_x2 = lambda x1,x2: norm(loc=x1**3,scale=0.1).logpdf(x2)

#logp of the joint
pi_tot = lambda x: pi_x1(x[0]) + pi_x2(x[0],x[1])

#%% sampling of the joint using emcee

#10 walkers
nwalkers = 10
#2 dimensions
ndim = 2

#steps to be taken
nsteps = 1000

#Initial state
initial_state = np.random.randn(nwalkers,ndim)

#sampler
sampler = emcee.EnsembleSampler(nwalkers, ndim, pi_tot)

#let the sampler sample
res = sampler.run_mcmc(initial_state, nsteps, progress = True)

#%%

#corner plot with some burning
corner(sampler.get_chain(flat=True,discard=500))

#%%

def iterate(n=5000,log_factor=10,scale=.75):
    
    
    #Initialize
    x1 = 0
    x2 = 0

    #Get logp for x1
    logp_x1 = pi_x1(x1)
    
    #Proposal dist
    proposal = norm(loc=0,scale=1)
    
    factor = 1

    for i in range(n):
        
        accept = False
               
        #Get uniform sample (slow)
        u = uniform.rvs()
        
        if log_factor: 
            factor = np.exp(uniform.rvs(loc=-log_factor,scale=2*log_factor))
        
    
        #Generate candidate 
        x1_cand = x1 + proposal.rvs() * scale * factor
        
        #Evaluate logp of candidate
        #Note that pi(x1|x2) \propto pi(x1,x2)/pi(x2) \propto pi(x1,x2)
        logp_x1_cand = pi_tot([x1_cand,x2])
        
        #If diff logp is greater than logu, accept
        if (logp_x1_cand - logp_x1) > np.log(u): 
            
            #Update x1
            x1 = x1_cand
        
            #And logp of x1
            logp_x1 = logp_x1_cand
            
            #accepted
            accept = True
        
        #Conditionally sample x2 (maybe slow)
        x2 = norm(loc=x1**3,scale=0.1).rvs()
        
        #Yield back samples
        yield x1, x2, logp_x1, accept

df = pd.DataFrame.from_records(
    iterate(scale=.75,log_factor=1)
    )

#%%

corner(df.iloc[500:,:2])

#%%

#Plot logp
plt.plot(df.iloc[:,-2])

#%% acceptance rate
df.iloc[:,-1].sum() / len(df)

#%%

plt.scatter(df.iloc[:,0],df.iloc[:,1],c=np.linspace(0,1,len(df)))