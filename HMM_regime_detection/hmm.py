import numpy as np
import pandas as pd
import scipy as sp



class Hmm:
    
    def __init__(self, N_hidden, N_obs, emission_model, pi, A=None):
        
        self.N_hidden = N_hidden
        self.N_obs = N_obs
        self.Bt = None  ## Emission probabilities of data. Depends on emission_model params

        self.emission = emission_model

        if A is not None:

            self.A = A
        
        else:
            
            A_rand = np.random.rand(N_hidden, N_hidden) 
            self.A = A_rand / A_rand.sum(axis=1, keepdims=True)   # normalize each row
        
        if pi is not None:
            self.pi = pi
        
        else:
            
            pi = np.random.rand((N_hidden)) 
            self.pi = pi / pi.sum()


    ''' 
        Methods
        
    '''

    def forward_coeff(self, data_obs):

        N_seq = data_obs.shape[0]
        T = data_obs.shape[1]
        
        alpha = np.zeros((N_seq, T, self.A.shape[0]))
        c = np.zeros((N_seq, T))
        

        #self.Bt = self.emission.emission_probability()  # return (N_seq, T, N_hidden)
                
        alpha_0 = self.pi[np.newaxis, :] * self.Bt[:, 0, :]
        #print(alpha_0.shape)
        
        c[:, 0] = 1 / (alpha_0.sum(axis=1))
        
        alpha[:, 0, :] = alpha_0[:, :] * c[:, 0, np.newaxis]
        
        
        for t in range(1, T):
    
            alpha_t = ((self.A @ (alpha[:, t-1, :].T)).T)*(self.Bt[:, t, :])
            
            c[:, t] = 1 / (alpha_t.sum(axis = 1))

            alpha[:, t, :] = alpha_t * c[:, t, np.newaxis]
        
        
        return alpha, c 

    def prob_obs(self, c):
        
        logP = -np.sum(np.log(c))
        
        return np.exp(logP)
        
    def prob_log(self, c):
        
        logP = - np.sum(np.log(c)) 
        
        return logP
    
    def backward_coeff(self, data_obs, c):

        N_seq = data_obs.shape[0]
        T = data_obs.shape[1]
        
        beta = np.zeros((N_seq, T, self.A.shape[0]))

        beta[:, -1, :] = 1 * c[:, -1, np.newaxis]
        
        #Bt = self.emission.emission_probability() 
        
        for t in range(1, T):
            
            beta_t = (self.A @ ((beta[:, -t, :] * self.Bt[:, -t, :]).T)).T

            beta[:, -1-t] = c[:, -1-t, np.newaxis] * beta_t
        
        
        return beta

    def update_A_B(self, data_obs):
        
        N_seq = data_obs.shape[0]
        T = data_obs.shape[1]
         
        gamma_ij = np.zeros((N_seq, T, self.A.shape[0], self.A.shape[0]))
        
        alpha, c = self.forward_coeff(data_obs)
        beta = self.backward_coeff(data_obs, c)

        gamma = alpha * beta
        
        b = self.Bt[:, 1:, :] # (N_seq, T, N_states)
        gamma_ij[:, :-1, :, :] = alpha[:, :-1, :, np.newaxis] * self.A[np.newaxis, np.newaxis, :, :] * (b[:, :, np.newaxis, :]) * beta[:, 1:, np.newaxis, :]

        A_prime = gamma_ij[:, :-1, :, :].sum(axis=(0, 1)) / (((gamma[:, :-1, :]/c[:, :-1, np.newaxis]).sum(axis=(0,1)))[:, np.newaxis])

        self.emission.update_emission(gamma, c)
        
        pi_prime = gamma[:, 0, :].sum(axis=(0, 1)) / gamma[:, 0, :].sum()

        return A_prime, pi_prime, c

    def Baum_Welch_algorithm(self, data_obs, N_max=100, eps=1e-8):

        step = 0
        err = 1
        #_, c = self.forward_coeff(data_obs)
        
        #P0_AB = self.prob_obs(c)
        err_list = []
        
        while step <= N_max:
            
            self.Bt = self.emission.emission_probability()  # return (N_seq, T, N_hidden)
            A_prime, pi_prime, c = self.update_A_B(data_obs)  

            self.A = A_prime
            self.pi = pi_prime.reshape((-1))

            err_list.append(self.compute_likelihood(data_obs, c))
  
            step +=1
        
        return err_list

    def compute_likelihood(self, obs_seq, c = None):

        if c is None:
            
            _, c = self.forward_coeff(obs_seq)
        
        log_likelihood = -np.sum(np.log(c), axis=1) / c.shape[1]
        
        return log_likelihood

    def Viterbi(self, data_obs):

        v = np.zeros((data_obs.shape[0], data_obs.shape[1], self.N_hidden))
        bt = np.zeros((data_obs.shape[0], data_obs.shape[1], self.N_hidden))
        self.Bt = self.emission.emission_probability()  # return (N_seq, T, N_hidden)
        v[:, 0, :] = self.pi * self.Bt[:, 0, :]
        bt[:, 0, :] = 0

        for t in range(1, data_obs.shape[1]):

            temp = ((self.A @ (v[:, t-1, :].T)).T)*(self.Bt[:, t, :]) 
            temp = temp / temp.sum(axis=1, keepdims=True)
            v[:, t, :] = np.max(temp, axis=1, keepdims=False)
            bt[:, t, :] = np.argmax(temp, axis=1, keepdims=False)
        
        
        self.decoded_probs = v[:, -1, 0]
        
        self.best_seqs = bt[:, :, 0]