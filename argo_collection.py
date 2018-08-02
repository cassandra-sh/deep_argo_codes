#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 09:28:08 2018

@author: cassandra
"""

class ArgoCollection:
    """
    A collection of ArgoFloat objects
    
    Has methods to load ArgoFloats (__init__) and produce new ArgoFloat
    objects which are interpolated between existing ArgoFloat objects
    
    The interpolation options are several:
        1. mean shallow, nearest deep
                basically, take the average of regular Argo floats in a region
                and take the deep argo floats' deep measurements as deep
    """
    
    def __init__(self):
        """
        
        """
        pass
    
    