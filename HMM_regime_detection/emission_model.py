import numpy as np
from abc import ABC, abstractmethod
from scipy.special import gamma as Gamma
from scipy.special import digamma
from scipy.optimize import brentq

class EmissionModel(ABC):

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
   
    def __init__(self, N_hidden, data_obs, N_obs, B):

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
   
    def __init__(self, N_hidden, data_obs, gauss_params=None):

        super().__init__(N_hidden, data_obs)
        
        self.gauss_params = np.zeros((N_hidden, 2))
        if gauss_params is None:
            self.gauss_params[:, 0] = np.random.rand(N_hidden)       # mu casuali
            self.gauss_params[:, 1] = abs(np.random.rand(N_hidden)) + 0.5 # sigma > 0

        else:
            self.gauss_params = gauss_params

    def Gaussian(self, x, mu, sigma):

        return np.exp(-(x-mu)**2/(2*sigma**2))/np.sqrt(2*np.pi*sigma**2)
    
    def emission_probability(self):
        
        Bt = self.Gaussian(self.data_obs[:, :, np.newaxis], self.gauss_params[np.newaxis, np.newaxis, :, 0], self.gauss_params[np.newaxis, np.newaxis, :, 1])
        
        return Bt

    def update_emission(self, gamma, c):
         
        self.gauss_params[:, 0] = (gamma[:, :, :] * self.data_obs[:, :, np.newaxis]/c[:, :, np.newaxis]).sum(axis=(0,1)) / (gamma[:, :, :]/c[:, :, np.newaxis]).sum(axis=(0,1))
        self.gauss_params[:, 1] = np.sqrt((gamma[:, :, :] * self.data_obs[:, :, np.newaxis]**2 /c[:, :, np.newaxis]).sum(axis=(0,1)) / (gamma[:, :, :]/c[:, :, np.newaxis]).sum(axis=(0,1)) - self.gauss_params[:, 0]**2)  


class Student_Emission(EmissionModel):
   
    def __init__(self, N_hidden, data_obs, student_params=None):

        super().__init__(N_hidden, data_obs)
        
        self.student_params = np.zeros((N_hidden, 3))
        if student_params is None:
            self.student_params[:, 0] = np.random.rand(N_hidden)       # mu casuali
            self.student_params[:, 1] = abs(np.random.rand(N_hidden)) + 1 # sigma > 0
            self.student_params[:, 2] = abs(np.random.rand(N_hidden)) + 3
        else:
            self.student_params = student_params

    def Student(self, x, mu, sigma, n):

        return (Gamma((n+1)/2) / (Gamma(n/2)*np.sqrt(n*np.pi)*sigma)) * (1+(1/n)*((x-mu)/sigma)**2)**(-(n+1)/2)
    
    def MLE_student(self, x, mu, sigma, n):
        z2 = ((x - mu) / sigma) ** 2
        return (0.5*digamma((n+1)/2) - 0.5*digamma(n/2)
                - 1/(2*n)
                - 0.5*np.log(1 + z2/n)
                + z2 * (n+1) * (1 + z2/n)**(-1) / (2 * n**2))
    
    def sum_MLE(self, n_guess, i, gamma, c):
        scores = self.MLE_student(
            self.data_obs[:, :],        # raw data, not weighted!
            self.student_params[i, 0],
            self.student_params[i, 1],
            n_guess)
        weights = gamma[:, :, i] / c[:, :]   # posterior for state i
        return np.sum(weights * scores)

    def emission_probability(self):
        
        Bt = self.Student(self.data_obs[:, :, np.newaxis], self.student_params[np.newaxis, np.newaxis, :, 0], self.student_params[np.newaxis, np.newaxis, :, 1], self.student_params[np.newaxis, np.newaxis, :, 2])
        
        return Bt

    def update_emission(self, gamma, c):
         
        self.student_params[:, 0] = (gamma[:, :, :] * self.data_obs[:, :, np.newaxis]/c[:, :, np.newaxis]).sum(axis=(0,1)) / (gamma[:, :, :]/c[:, :, np.newaxis]).sum(axis=(0,1))
        self.student_params[:, 1] = np.sqrt((gamma[:, :, :] * self.data_obs[:, :, np.newaxis]**2 /c[:, :, np.newaxis]).sum(axis=(0,1)) / (gamma[:, :, :]/c[:, :, np.newaxis]).sum(axis=(0,1)) - self.student_params[:, 0]**2)  
        
        self.update_nu(gamma, c)

    def update_nu(self,gamma, c, nmin = 3, nmax = 10000):

        for i in range(self.N_hidden):
            
            #print(f"state {i}: f(nmin)={self.sum_MLE(nmin, i, gamma, c):.4f}, f(nmax)={self.sum_MLE(nmax, i, gamma, c):.4f}")
            #self.student_params[i, 2] = brentq(lambda n_guess: self.sum_MLE(n_guess, i, gamma, c), nmin , nmax)
        
            try:
                self.student_params[i, 2] = brentq(lambda n_guess: self.sum_MLE(n_guess, i, gamma, c), nmin, nmax)
            except ValueError:
                self.student_params[i, 2] = nmax  # cap at nmax, effectively Gaussian


