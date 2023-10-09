#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 13:52:49 2022

@author: gustav
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


#%%

class logit(BaseEstimator,TransformerMixin):
    
    def __init__(self,**kwargs):
        
        self.Xmin = kwargs.get('Xmin')
        self.Xmax = kwargs.get('Xmax') 
        
        self.tol = kwargs.get('tol',1e-10)
        
    def fit(self,X,y=None):
            
        Xmin = X.min(axis=0)
        Xmax = X.max(axis=0)
                
        self.Xmin = Xmin
        self.Xmax = Xmax
        
        return self
    
    
    def transform(self,X,y=None):
        
        tol = self.tol
        
        Xmin = self.Xmin - tol
        Xmax = self.Xmax + tol
        
        #Scale between 0 and 1
        Xtrans = (X - Xmin) / (Xmax - Xmin)
        
        #Calculate transform
        Xtrans = np.log(Xtrans/(1-Xtrans)) 
        
        #Return transform
        return Xtrans
        
    def inverse_transform(self,Xtrans,y=None):
        
        Xmin = self.Xmin
        Xmax = self.Xmax
        
        #Get value between 0 and 1 
        X = np.exp(Xtrans) / (np.exp(Xtrans) + 1)
        
        #Scale back
        X = X * (Xmax - Xmin) + Xmin
        
        #Return inverse transformed
        return X
    
    
            
if __name__ == '__main__':
    
    import matplotlib.pyplot as plt 
    
    # trans = range_tanh(Xmax=np.array([40]),Xmin=np.array([1/40]),
                        # alpha=1/2,eps_range=1.05)
    
    trans = logit(Xmax=np.array([40]),Xmin=np.array([1/40]))
    
    X = np.linspace(1/40,40,100000)
    Xtrans = trans.transform(X)
    X2 = trans.inverse_transform(Xtrans)
    
    plt.plot(X,Xtrans)
    
