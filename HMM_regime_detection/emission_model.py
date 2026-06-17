import numpy as np
from abc import ABC, abstractmethod
from scipy.special import gamma as Gamma
from scipy.special import gammaln
from scipy.special import digamma
from scipy.optimize import brentq

class EmissionModel(ABC):

    '''
        Parent class of Emission models for the Hmm class
            
            - N_hidden : number of hidden states of the model;
            - data_obs: observation sequences of length T. np.array of shape (N_seq, T) 

        Two parent methods:

            - emission_probability : computes the emission probability of hidden states given emission parameters
            - update_emission : called in the maximization step of the expectation-maximization algorithm, to update the emission parameters

    '''

    def __init__(self, N_hidden, data_obs):

        self.N_hidden = N_hidden
        self.data_obs = data_obs


    @abstractmethod
    def emission_probability(self):
      pass

    @abstractmethod
    def update_emission(self, gamma, c):
      pass


class Discrete_Emission(EmissionModel):

    '''

        Emission model for discrete observations
           
    '''
   
    def __init__(self, N_hidden, data_obs, N_obs, B):

        '''

            N_obs : number of discrete outcomes of observations
            B : emission probability matrix

        '''

        super().__init__(N_hidden, data_obs)
        
        self.N_obs = N_obs
        
        if B is not None:
            
            self.B = B
        
        else:
            
            B_rand = np.random.rand(N_hidden, N_obs) 
            self.B = B_rand / B_rand.sum(axis=1, keepdims=True)   # normalize each row
        


    
    def emission_probability(self):
        
        Bt = self.B[:, self.data_obs]  # shape: (D, N_seq, T)
        Bt = Bt.transpose(1, 2, 0)  # shape: (N_seq, T, D)
        
        return Bt

    def update_emission(self, gamma, c):

        for k in range(self.N_obs):
            
            temp_idx = self.data_obs[:, :] == k
            
            self.B[:, k] = (np.ndarray.flatten(((gamma[:, :, :]/c[:, :, np.newaxis])*temp_idx[:, :, np.newaxis]).sum(axis=(0, 1), keepdims=True).T/ (gamma[:, :, :]/c[:, :, np.newaxis]).sum(axis=(0, 1), keepdims=True).T))
        
        
   
class Gaussian_Emission(EmissionModel):


    '''

        Emission model for discrete observations
           
    '''
   
    def __init__(self, N_hidden, data_obs, params=None):


        '''

            Emission model for continuous Gaussian emissions

                - the emission parameters of each hidden states are the mean and the variance
           
        '''

        super().__init__(N_hidden, data_obs)
        
        self.params = np.zeros((N_hidden, 2))
        if params is None:
            self.params[:, 0] = np.random.rand(N_hidden)       # mu casuali
            self.params[:, 1] = abs(np.random.rand(N_hidden)) + 0.5 # sigma > 0

        else:
            self.params = params

    def Gaussian(self, x, mu, sigma):

        return np.exp(-(x-mu)**2/(2*sigma**2))/np.sqrt(2*np.pi*sigma**2)
    
    def emission_probability(self):
        
        Bt = self.Gaussian(self.data_obs[:, :, np.newaxis], self.params[np.newaxis, np.newaxis, :, 0], self.params[np.newaxis, np.newaxis, :, 1])
        
        return Bt

    def update_emission(self, gamma, c):

        '''
            Perform maximization using the closed formula from the log-likelihood derivatives

        '''
         
        self.params[:, 0] = (gamma[:, :, :] * self.data_obs[:, :, np.newaxis]/c[:, :, np.newaxis]).sum(axis=(0,1)) / (gamma[:, :, :]/c[:, :, np.newaxis]).sum(axis=(0,1))
        self.params[:, 1] = np.sqrt((gamma[:, :, :] * self.data_obs[:, :, np.newaxis]**2 /c[:, :, np.newaxis]).sum(axis=(0,1)) / (gamma[:, :, :]/c[:, :, np.newaxis]).sum(axis=(0,1)) - self.params[:, 0]**2)  


class Student_Emission(EmissionModel):

    '''

        Emission model for the Student distribution

        Emission parameters of each hidden states are the mean, variance and degree of freedom nu.

           
    '''
   
    def __init__(self, N_hidden, data_obs, student_params=None, count_max = 25, tol_maximization=1e-5, nu_start=3, alpha=1, params_to_optimize = [True, True, True]):

        '''
            count_max : maximum number of inner iterations in the maximization step to find mu, nu and sigma. Default to 10
            tol_maximization: convergence tolerance in parameters for the inner iterations
            nu_count : update nu every nu_count iteration
            converged : (bool) indicates whether parameters update in the inner EM algorithm converged
        '''
        super().__init__(N_hidden, data_obs)
        
        self.params = np.zeros((N_hidden, 3))
        self.count_max = count_max
        self.tol_maximization = tol_maximization
        self.nu_start = nu_start
        self.param_list_learning = []
        self.params_to_optimize =params_to_optimize
        
        self.converged = False
        self.alpha = alpha #mixing parameter
        if student_params is None:
            self.params[:, 0] = np.random.rand(N_hidden)       # mu casuali
            self.params[:, 1] = abs(np.random.rand(N_hidden)) + 1 # sigma > 0
            self.params[:, 2] = abs(np.random.rand(N_hidden)) + 3

        else:
            self.params = student_params
        
        

    def Student(self, x, mu, sigma, n):
        # Calculate the log of the PDF components
        log_pdf = (gammaln((n + 1) / 2) 
                - gammaln(n / 2) 
                - 0.5 * np.log(n * np.pi) 
                - np.log(sigma)
                - ((n + 1) / 2) * np.log1p((1 / n) * ((x - mu) / sigma)**2))
        
        return np.exp(log_pdf)
    
    def MLE_student(self, x, mu, sigma, n):

        '''
            Derivative of Log likelihood of the Student distribution w.r.t. nu; find its zeros with the brentq method 
        
        '''
        z2 = ((x - mu) / sigma) ** 2
        return (0.5*digamma((n+1)/2) - 0.5*digamma(n/2)
                - 1/(2*n)
                - 0.5*np.log(1 + z2/n)
                +0.5*(n+1)*z2*((1+z2/n)**(-1))/(n**2))
    
    def sum_MLE(self, mu, sigma, n_guess, i, gamma, c):
        
        scores = self.MLE_student(
            self.data_obs[:, :],        # raw data, not weighted!
            mu,
            sigma,
            n_guess)
        
        weights = gamma[:, :, i] / c[:, :]   # posterior for state i
        
        return np.sum(weights * scores)

    def emission_probability(self):
        
        Bt = self.Student(self.data_obs[:, :, np.newaxis], self.params[np.newaxis, np.newaxis, :, 0], self.params[np.newaxis, np.newaxis, :, 1], self.params[np.newaxis, np.newaxis, :, 2])
        
        return Bt
    
    def update_emission(self, gamma, c):
        
        '''
            The update emission is performed with an inner expectation-maximization algorithm.
            
        '''
        count = 0
        
        while True:
            
            weights = self.compute_weights(gamma, c)  # fixed for entire M-step    
            mu_old = self.params[:, 0].copy()
            sigma_old = self.params[:, 1].copy()
            nu_old = self.params[:, 2].copy()

            if self.params_to_optimize[0] == True:
                self.params[:, 0] = (np.sum(weights * self.data_obs[:, :, np.newaxis], axis=(0,1))
                                   / np.sum(weights, axis=(0,1)))

            weights = self.compute_weights(gamma, c)

            if self.params_to_optimize[1] == True:
                self.params[:, 1] = np.sqrt(
        
                    np.sum(weights * (self.data_obs[:, :, np.newaxis] - self.params[np.newaxis, np.newaxis,:, 0])**2, axis=(0,1)) / np.sum(weights, axis=(0,1))
        
                )
        
            
            if self.params_to_optimize[2] == True and count > self.nu_start:
                
                for i in range(self.N_hidden):
                
                    self.params[i, 2] = self.alpha*self.safe_solve_nu(self.params[i, 0], self.params[i, 1], i, gamma, c) + (1-self.alpha)*nu_old[i]
            
            

             # --- convergence: max relative change across all params and states ---
            delta_mu    = np.max(np.abs(self.params[:, 0] - mu_old)    / (np.abs(mu_old)    + 1e-8))
            delta_sigma = np.max(np.abs(self.params[:, 1] - sigma_old) / (np.abs(sigma_old) + 1e-8))
            
            delta_nu = np.max(np.exp(-nu_old / 30) * np.abs(self.params[:, 2] - nu_old) / (np.abs(nu_old) + 1e-8)) # empirical value of 20: above 20 we consider the student distribution equivalent to the Gaussian   

            count+=1

            if max(delta_mu, delta_sigma, delta_nu) <= self.tol_maximization: 
                
                self.converged = True
                '''
                print('#####################################################################################################################################')  
                print('End Student maximization: achieved accuracy of', self.tol_maximization)
                '''
                break

            elif count>self.count_max:
                
                self.converged = False
                '''
                print('#####################################################################################################################################')  
                print('End Student maximization: maximum number reached')
                '''
                break
            
        self.param_list_learning.append(self.params.copy())

    def compute_weights(self, gamma, c):

        weights = np.zeros((self.data_obs.shape[0], self.data_obs.shape[1], self.N_hidden))

        weights[:, :, :] = (gamma[:, :, :]/ c[:, :, np.newaxis]) * (self.params[np.newaxis, np.newaxis, :, 2]+1) / (self.params[np.newaxis, np.newaxis, :, 2] + (self.data_obs[:, :, np.newaxis]-self.params[np.newaxis, np.newaxis, :, 0])**2 / (self.params[np.newaxis, np.newaxis, :, 1]**2 +1e-8))
        
        return weights
    
    def safe_solve_nu(self, mu, sigma, i, gamma, c):
        # Define the objective function for the root finder
        f = lambda nu: self.sum_MLE(mu, sigma, nu, i, gamma, c)
        
        a, b = 3.001, 50.0
        fa, fb = f(a), f(b)
        
        # Check if signs are the same at boundaries
        if fa * fb > 0:
            # If both negative, the root is likely > 600 (Gaussian)
            
            if fa<0:
                return b
            # If both positive, the root is likely < 3 (Very heavy tails)
            else:
                return a
                
        return brentq(f, a, b, xtol=1e-12)
    




