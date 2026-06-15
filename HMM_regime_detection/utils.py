import numpy as np
import pandas as pd
import scipy as sp
from scipy.special import gamma as Gamma
from scipy.special import gammaln
from scipy.special import digamma
import matplotlib.pyplot as plt

# Probability Distributions

def Student(x, mu, sigma, n):
    # Calculate the log of the PDF components
    log_pdf = (gammaln((n + 1) / 2) 
            - gammaln(n / 2) 
            - 0.5 * np.log(n * np.pi) 
            - np.log(sigma)
            - ((n + 1) / 2) * np.log1p((1 / n) * ((x - mu) / sigma)**2))
    
    return np.exp(log_pdf)

def Gaussian(x, mu, sigma):

    return np.exp(-0.5*((x-mu)/sigma)**2)/(np.sqrt(2*np.pi*sigma**2))


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


def plot_regimes(dates,prices, hidden_states, n_states, colors):
    
    
    fig, ax = plt.subplots(figsize=(14, 5))
    
    ax.plot(dates, prices, color='black', linewidth=2, zorder=2)
    
    
    i = 0
    
    while i < len(hidden_states):
        
        state = int(hidden_states[i])
        j = i
       
        while j < len(hidden_states) and hidden_states[j] == state:
            j += 1
        ax.axvspan(dates[i], dates[min(j, len(dates)-1)], 
                   alpha=0.3, color=colors[state], zorder=1)
        i = j
    
    
    from matplotlib.patches import Patch
    legend = [Patch(facecolor=colors[k], alpha=0.3, label=f'Regime {k}') 
              for k in range(n_states)]
    ax.legend(handles=legend, loc='upper left')
    
    ax.set_xlabel('time')
    ax.set_ylabel('return')
    plt.tight_layout()
    plt.show()
