import numpy as np
import pandas as pd
import scipy as sp
from scipy.special import gamma as Gamma
from scipy.special import gammaln
from scipy.special import digamma

# Probability Distributions

def Student(x, mu, sigma, n):
    # Calculate the log of the PDF components
    log_pdf = (gammaln((n + 1) / 2) 
            - gammaln(n / 2) 
            - 0.5 * np.log(n * np.pi) 
            - np.log(sigma)
            - ((n + 1) / 2) * np.log1p((1 / n) * ((x - mu) / sigma)**2))
    
    return np.exp(log_pdf)


# Accept-reject method

def accept_reject(f, params, fmax=None, xmin=-1, xmax=1, resolution=100000):

    if fmax == None:
        
        x_lin= np.linspace(xmin,xmax, resolution)
        f_lin= f(x_lin, *params)
        fmax= np.max(f_lin)

    while True:

        x = np.random.uniform(xmin, xmax)

        y = f(x, *params)

        if np.random.uniform(0, fmax) < y:
            
            break
    
    return x

