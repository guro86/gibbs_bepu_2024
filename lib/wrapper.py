#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 16:01:14 2023

@author: gustav
"""

import numpy as np

class model_wrapper():
    
    
    def __init__(self,**kwargs):
        
        self.default = kwargs.get(
            'default',
            np.ones(5)
            )
        
        self.pos = kwargs.get('pos',[0,1,2])
        self.model = kwargs.get('model',None)
    
    def predict(self,x):
        
        default = self.default
        pos = self.pos
      
        X = default
        
        X[pos] = x
        
        return self.model.predict_fast(X[None,:])