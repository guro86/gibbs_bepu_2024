#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 15:41:18 2023

@author: gustav
"""

from scipy.stats import norm, truncnorm, uniform
import pickle
import numpy as np
import matplotlib.pyplot as plt
from lib import model_wrapper
from tqdm import tqdm
import pandas as pd
from corner import corner
from sklearn.preprocessing import StandardScaler
from lib.transforms import logit

# import warnings
# warnings.filterwarnings("error")

with open('gp_fgr.p','rb') as f:
    fgr_data, gp = pickle.load(f)

#%%

scaler = logit()

scaler.Xmax = np.array([40,10,1,1,10])
scaler.Xmin = np.array([.1,.1,.1,0,.1])

#Measurement vector
meas_v = fgr_data.meas_v

#Number of samples to generate
nsamp = 500

#Dimensions and number of experiments
nexp = 31

#Dimensions
dims_active = np.array([0,1,2])

ndims = 5

ndims_active = 3

#Mean and sigmas
mu = np.zeros(3)
sig = np.ones(3)


default = np.array([0.9,4,0.65,0.42,2.2])

#Local parameterss
X = scaler.transform(np.ones((nexp,ndims))*default)

R = lambda m: (0.1**2*m**2)**.5

#Likelihoods
likes = [norm(loc=m,scale=R(m)) for m in fgr_data.meas_v]

#Proposal 
propose = norm(scale=0.1)

propose_mu = norm(scale=0.1)
propose_sig = norm(scale=0.1)



#%%

def update_dim(X,d,e,mu,sig):
    
    hier = norm(loc=mu,scale=sig)
    
    pred = gp.gps[e].predict(
        scaler.inverse_transform(X[None,:])
                                 ).flatten()
    
    logp = likes[e].logpdf(pred) + hier.logpdf(X[dims_active]).sum()
    
    X_cand = np.empty(len(X))
    X_cand[:] = X[:]
    X_cand[d] += propose.rvs()
    
    pred_cand = gp.gps[e].predict(
        scaler.inverse_transform(X_cand[None,:])
        ).flatten()
    
    logp_cand = likes[e].logpdf(pred_cand) + hier.logpdf(X_cand[dims_active]).sum()
    
    u = uniform.rvs()
    
    if logp_cand - logp > np.log(u):

        X[:] = X_cand[:]
        
    return X

def update_hyper(X,mu,sig):
    
    mu_cand = mu + propose_mu.rvs()
    
    logp = norm.logpdf(X,loc=mu,scale=sig).sum()
    
    logp_cand = norm.logpdf(X,loc=mu_cand,scale=sig).sum()
    
    u = uniform.rvs()
    
    if logp_cand - logp > np.log(u):
        mu = mu_cand
        logp = logp_cand
        
    sig_cand = sig + propose_sig.rvs()
    
    logp_cand = norm.logpdf(X,loc=mu,scale=sig_cand).sum()
    
    u = uniform.rvs()
    
    if logp_cand - logp > np.log(u):
        sig = sig_cand
        logp = logp_cand
        
    
    return mu, sig
    
#%%
Xs = np.empty((nsamp,nexp,ndims))
sigs = np.empty((nsamp,ndims_active))
mus = np.empty((nsamp,ndims_active))

#Loop over samples
for s in tqdm(range(nsamp)):

    #Loop dimensions
    for d in dims_active:
    
        #Loop over experiments
        for e in range(nexp):
                    
            #Update the current dimension
            X[e,:] = update_dim(X=X[e],d=d,e=e,mu=mu,sig=sig)
            
        #Update hyper-parameters
        mu[d], sig[d] = update_hyper(X=X[:,d],mu=mu[d],sig=sig[d])
        
    Xs[s] = X
    mus[s] = mu
    sigs[s] = sig
    
#%%

Xp = scaler.transform(np.ones((500,ndims))*default)

Xp[:,:3] = norm(loc=mus,scale=sigs).rvs((500,3))
Xp = scaler.inverse_transform(Xp)

#%%

names = fgr_data.Xtrain.columns[:3]

corner(
       pd.DataFrame(
           Xp[:,:3],
           columns=names
           )
       )

plt.savefig('marg-post.pdf')

#%%

names = fgr_data.Xtrain.columns[:3]

names = [r'$\mu_{{{}}}$'.format(name) for name in fgr_data.Xtrain.columns[:3]]

corner(
       pd.DataFrame(
           mus,
           columns=names
           )
       )

plt.savefig('mu-post.pdf')


#%%

names = ['$\sigma\_{{{}}}$'.format(name) for name in fgr_data.Xtrain.columns[:3]]

corner(
       pd.DataFrame(
           sigs,
           columns=names
           )
       )

plt.savefig('sig-post.pdf')


#%%

pred_p = gp.predict(Xp)
mean = pred_p.mean(axis=0)
std = pred_p.std(axis=0)

pred_pp = pred_p + norm(scale=R(fgr_data.meas_v)).rvs((500,31))


q = [0.05, 0.95]

pred_q = np.quantile(pred_p, q,axis=0)
pred_qp = np.quantile(pred_pp, q,axis=0)

l = np.linspace(0,.5,2)

yerr = np.abs(pred_q-mean)
yerr_p = np.abs(pred_qp-mean)

# plt.plot(fgr_data.meas_v,mean,'o')

plt.errorbar(
    fgr_data.meas_v,
    mean,
    yerr,
    fmt='o'
      )

plt.plot(
    fgr_data.meas_v,
    mean,
    'o'
    )

plt.plot(l,l,'--')

plt.xlabel('Measured fission gas release [-]')
plt.ylabel('Predicted fission gas release [-]')

plt.savefig('validation.pdf')

#%%
plt.plot(
    fgr_data.meas_v,
    pred_qp[1],
    'o'
    )

plt.plot(l,l,'--')

plt.xlabel('Measured fission gas release [-]')
plt.ylabel('Predicted fission gas release [-]')

plt.savefig('ub.pdf')


#%%
plt.plot(
    fgr_data.meas_v,
    pred_qp[0],
    'o'
    )

plt.plot(l,l,'--')

plt.xlabel('Measured fission gas release [-]')
plt.ylabel('Predicted fission gas release [-]')

plt.savefig('lb.pdf')

