# -*- coding: utf-8 -*-
"""
@authors: Kaixin and Suzan  

Functions that determine the status of the cells based on the given probabilities in the main code.
"""
import random

def initial(seq,prob):
    """
    Generate initial state of the cells.
    """
    x = random.uniform(0, 1)
    if x<=prob[0]:
        y=seq[0]
    elif x<=prob[1]:
        z=random.randint(0,1)
        y=seq[1+z]
    elif x<=prob[2]:
        z=random.randint(0,4)
        y=seq[3+z]	
    return y
    
def ProbCatchF(seq,prob):
    """
    Generate whether or not a susceptible individual will be infected.
    """
    x = random.uniform(0, 1)
    if x<=prob[0]:
        y=seq[1]
    else:  
        y=seq[0]
    return y
    
def ProBeSusceptibleF(seq,prob):
    """
    Generate whether or not an immune individual will become susceptible.
    """
    x = random.uniform(0, 1)
    if x<=prob[0]:
        y=seq[1]
    else:  
        y=seq[0]
    return y
