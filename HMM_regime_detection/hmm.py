import numpy as np
import pandas as pd
import scipy as sp



class Hmm:
    
    def __init__(self, N_hidden, emission_model, pi=None, A=None, eps=1e-7):
        
        '''
            Initialization of the discrete hidden state Markov Model. Parameters:

                - N_hidden : number of hidden states of the model;
                - emission_model : object EmissionModel as defined in emission_model.py
                - pi : initial probablities of hidden states
                - A : transition matrix
        
        '''

        self.N_hidden = N_hidden
        self.Bt = None  ## Emission probabilities of data. Depends on emission_model params
        self.emission = emission_model
        self.eps = eps #tolerance for HMM EM algorithm
        
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

        '''
            
            Scaled forward algorithm to compute forward variables alpha_j(t). 
            alpha_j(t) are the probabilities of having the set of observations O(t1)O(t2)...O(t), with hidden state h(t) = j at time t, given the emission parameters. 

                - data_obs: observation sequences of length T. np.array of shape (N_seq, T)
        
        '''

        N_seq = data_obs.shape[0]
        T = data_obs.shape[1]
        
        alpha = np.zeros((N_seq, T, self.A.shape[0]))
        c = np.zeros((N_seq, T)) # scaling weight
        
        alpha_0 = self.pi[np.newaxis, :] * self.Bt[:, 0, :]
        
        c[:, 0] = 1 / (alpha_0.sum(axis=1)+1e-100)
        
        alpha[:, 0, :] = alpha_0[:, :] * c[:, 0, np.newaxis]
        
        
        for t in range(1, T):
            
            alpha_t = ((self.A.T @ (alpha[:, t-1, :].T)).T) * (self.Bt[:, t, :])
            
            c[:, t] = 1 / (alpha_t.sum(axis = 1)+1e-100)

            alpha[:, t, :] = alpha_t * c[:, t, np.newaxis]
        
        
        return alpha, c 

    def backward_coeff(self, data_obs, c):

        '''  
          
            Scaled backward algorithm to compute beta_j(t). 
            beta_j(t) are the probabilities of having the set of observations O(t+1)O(t+2)...O(T) and hidden state h(t)=j at time t, given the emission parameters. 

                - data_obs: observation sequences of length T. np.array of shape (N_seq, T)
                - c : scaling weights computed in the forward algorithm        
        
        '''

        N_seq = data_obs.shape[0]
        T = data_obs.shape[1]
        
        beta = np.zeros((N_seq, T, self.A.shape[0]))

        beta[:, -1, :] = 1 * c[:, -1, np.newaxis]
         
        for t in range(1, T):
            
            beta_t = (self.A @ ((beta[:, -t, :] * self.Bt[:, -t, :]).T)).T

            beta[:, -1-t] = c[:, -1-t, np.newaxis] * beta_t
        
        
        return beta
      

    def e_m_algorithm(self, data_obs):
        
        '''  
          
            Iteration of the expectation-maximization (Baum-Welch) algorithm. 
                
                - data_obs: observation sequences of length T. np.array of shape (N_seq, T)   
        
        '''

        N_seq = data_obs.shape[0]
        T = data_obs.shape[1]
         
        # expectation step

        eps_ij = np.zeros((N_seq, T, self.A.shape[0], self.A.shape[0])) # probability of hidden states h(t) = i and h(t+1) = j, given observation sequence O(1)...O(T) and emission paramenters
        
        alpha, c = self.forward_coeff(data_obs)
        beta = self.backward_coeff(data_obs, c)

        gamma = alpha * beta # probabilities of each hidden states at all times P(h(t)= i | O,params)  --> equivalent to mixture weights in unsupervised learning for clustering
        eps_ij[:, :-1, :, :] = alpha[:, :-1, :, np.newaxis] * self.A[np.newaxis, np.newaxis, :, :] * (self.Bt[:, 1:, np.newaxis, :]) * beta[:, 1:, np.newaxis, :]

        likelihood = self.compute_likelihood(data_obs, c)

        ###################################################################################################################################################################################################################
        
        # maximization step : update transition matrix A and emission parameters, calling the method update_emission of the EmissionModel object

        A_prime = eps_ij[:, :-1, :, :].sum(axis=(0, 1)) / (((gamma[:, :-1, :]/(c[:, :-1, np.newaxis])).sum(axis=(0,1)))[:, np.newaxis])
        self.emission.update_emission(gamma, c)
        
        pi_prime = gamma[:, 0, :].sum(axis=(0, 1)) / gamma[:, 0, :].sum()

        return A_prime, pi_prime, c, likelihood

    def Baum_Welch(self, data_obs, data_test= None, N_max=100):

        '''  
          
            Baum-Welch cycle to learn the HMM parameters
                
                - data_obs: observation sequences of length T. np.array of shape (N_seq, T)   
                - N_max : maximum number of iterations
                - eps : convergence threshold

        '''

        step = 0
        
        print('############################################################################################################################################################################')
        print('Start learning: Baum-Welch expectation - maximization algorithm ')
        
        err_list = []
        err_diff = 1
        
        
        while step <= N_max and abs(err_diff) > self.eps:
            

            self.Bt = self.emission.emission_probability()  # return (N_seq, T, N_hidden)
            A_prime, pi_prime, _,likelihood = self.e_m_algorithm(data_obs)  
            
            self.A = A_prime
            self.pi = pi_prime.reshape((-1))

            err_list.append(-likelihood)
            
            if step >= 1:
                err_diff = ((err_list[-1] - err_list[-2])/err_list[-2])
            
            step +=1
            
            if data_test is not None:

                
                likelihood_test = self.compute_likelihood(data_test)
                print(f"Iteration {step:>4d}  |  -logL = {float(err_list[-1]):>14.6f}  |  -ΔL/L = {float(err_diff)*100:.6f}% |  -logL_test = {float(likelihood_test):.6f}%")
            
            else:
                
                print(f"Iteration {step:>4d}  |  -logL = {float(err_list[-1]):>14.6f}  |  -ΔL/L = {float(err_diff)*100:.6f}%")
        
        print('Learning Finished')
        print('############################################################################################################################################################################')

        return err_list

    def compute_likelihood(self, obs_seq, c = None):

        '''
            

            Compute the log-likelihood of the observation sequence obs_seq given the parameters 

        '''

        if c is None:
            
            _, c = self.forward_coeff(obs_seq)
        
        log_likelihood = -np.sum(np.log(c), axis=1) / c.shape[1]
        
        return log_likelihood

    def Viterbi(self, data_obs):

        '''
            Implementation of the Viterbi (Decoding) algorithm: it finds the most probaable sequence of hidden state given observations data_obs O(1)...O(T) and model parameters

        '''

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

        return self.best_seqs
    

    def prob_obs(self, c):
        
        logP = -np.sum(np.log(c))
        
        return np.exp(logP)
        
    def prob_log(self, c):
        
        logP = - np.sum(np.log(c)) 
        
        return logP