'''
Module implementing Hidden Markov model parameter estimation.

To avoid repeated warnings of the form "Warning: divide by zero encountered in log", 
it is recommended that you use the command "np.seterr(divide="ignore")" before 
invoking methods in this module.  This warning arises from the code using the 
fact that python sets log 0 to "-inf", to keep the code simple.

Initial version created on Mar 28, 2012

@author: kroon, herbst
'''

from warnings import warn
import numpy as np
from gaussian import Gaussian
np.seterr(divide="ignore")

class HMM(object):
    '''
    Class for representing and using hidden Markov models.
    Currently, this class only supports left-to-right topologies and Gaussian
    emission densities.

    The HMM is defined for n_states emitting states (i.e. states with 
    observational pdf's attached), and an initial and final non-emitting state (with no 
    pdf's attached). The emitting states always use indices 0 to (n_states-1) in the code.
    Indices -1 and n_states are used for the non-emitting states (-1 for the initial and
    n_state for the terminal non-emitting state). Note that the number of emitting states
    may change due to unused states being removed from the model during model inference.

    To use this class, first initialize the class, then either use load() to initialize the
    transition table and emission densities, or fit() to initialize these by fitting to
    provided data.  Once the model has been fitted, one can use viterbi() for inferring
    hidden state sequences, forward() to compute the likelihood of signals, score() to
    calculate likelihoods for observation-state pairs, and sample()
    to generate samples from the model.
        
    Attributes:
    -----------
    data : (d,n_obs) ndarray 
        An array of the trainining data, consisting of several different
        sequences.  Thus: Each observation has d features, and there are a total of n_obs
        observation.   An alternative view of this data is in the attribute signals.

    diagcov: boolean
        Indicates whether the Gaussians emission densities returned by training
        should have diagonal covariance matrices or not.
        diagcov = True, estimates diagonal covariance matrix
        diagcov = False, estimates full covariance matrix

    dists: (n_states,) list
        A list of Gaussian objects defining the emitting pdf's, one object for each 
        emitting state.

    maxiters: int
        Maximum number of iterations used in Viterbi re-estimation.
        A warning is issued if 'maxiters' is exceeded. 

    rtol: float
        Error tolerance for Viterbi re-estimation.
        Threshold of estimated relative error in log-likelihood (LL).

    signals : ((d, n_obs_i),) list
        List of the different observation sequences used to train the HMM. 
        'd' is the dimension of each observation.
        'n_obs_i' is the number of observations in the i-th sequence.
        An alternative view of thise data is in the attribute data.
            
    trans : (n_states+1,n_states+1) ndarray
        The left-to-right transition probability table.  The rightmost column contains probability
        of transitioning to final state, and the last row the initial state's
        transition probabilities.   Note that all the rows need to add to 1. 
    
    Methods:
    --------
    fit():
        Fit an HMM model to provided data using Viterbi re-estimation (i.e. the EM algorithm).

    forward():
        Calculate the log-likelihood of the provided observation.

    load():
        Initialize an HMM model with a provided transition matrix and emission densities
    
    sample():
        Generate samples from the HMM
    
    viterbi():
        Calculate the optimal state sequence for the given observation 
        sequence and given HMM model.
    
    Example (execute the class to run the example as a doctest)
    -----------------------------------------------------------
    >>> import numpy as np
    >>> from gaussian import Gaussian
    >>> signal1 = np.array([[ 1. ,  1.1,  0.9, 1.0, 0.0,  0.2,  0.1,  0.3,  3.4,  3.6,  3.5]])
    >>> signal2 = np.array([[0.8, 1.2, 0.4, 0.2, 0.15, 2.8, 3.6]])
    >>> data = np.hstack([signal1, signal2])
    >>> lengths = [11, 7]
    >>> hmm = HMM()
    >>> hmm.fit(data,lengths, 3)
    >>> trans, dists = hmm.trans, hmm.dists
    >>> means = [d.get_mean() for d in dists]
    >>> covs = [d.get_cov() for d in dists]
    >>> covs = np.array(covs).flatten()
    >>> means = np.array(means).flatten()
    >>> print(trans)
    [[0.66666667 0.33333333 0.         0.        ]
     [0.         0.71428571 0.28571429 0.        ]
     [0.         0.         0.6        0.4       ]
     [1.         0.         0.         0.        ]]
    >>> print(covs)
    [0.01666667 0.01459184 0.0896    ]
    >>> print(means)
    [1.         0.19285714 3.38      ]
    >>> signal = np.array([[ 0.9515792,   0.9832767,   1.04633007,  1.01464327,  0.98207072, 1.01116689, 0.31622856,  0.20819263,  3.57707616]])
    >>> vals, ll = hmm.viterbi(signal)
    >>> print(vals)  
    [0 0 0 0 0 0 1 1 2]
    >>> print("%.8f" % ll)
    2.90534116
    >>> hmm.load(trans, dists)
    >>> vals, ll = hmm.viterbi(signal)
    >>> print(vals)
    [0 0 0 0 0 0 1 1 2]
    >>> print("%.8f" % ll)
    2.90534116
    >>> print(hmm.score(signal, vals))
    2.905341164334513
    >>> print("%.8f" % hmm.forward(signal))
    2.90534236
    >>> signal = np.array([[ 0.9515792,   0.832767,   3.57707616]])
    >>> vals, ll = hmm.viterbi(signal)
    >>> print(vals)
    [0 1 2]
    >>> print(ll)
    -14.975826945102282
    >>> samples, states = hmm.sample()
    '''

    def __init__(self, diagcov=True, maxiters=20, rtol=1e-4): 
        '''
        Create an instance of the HMM class, with n_states hidden emitting states.
        
        Parameters
        ----------
        diagcov: boolean
            Indicates whether the Gaussians emission densities returned by training
            should have diagonal covariance matrices or not.
            diagcov = True, estimates diagonal covariance matrix
            diagcov = False, estimates full covariance matrix

        maxiters: int
            Maximum number of iterations used in Viterbi re-estimation
            Default: maxiters=20

        rtol: float
            Error tolerance for Viterbi re-estimation
            Default: rtol = 1e-4
        '''
        
        self.diagcov = diagcov
        self.maxiters = maxiters
        self.rtol = rtol
        
    def fit(self, data, lengths, n_states):
        '''
        Fit a left-to-right HMM model to the training data provided in `data`.
        The training data consists of l different observation sequences, 
        each sequence of length n_obs_i specified in `lengths`. 
        The fitting uses Viterbi re-estimation (an EM algorithm).

        Parameters
        ----------
        data : (d,n_obs) ndarray 
            An array of the training data, consisting of several different
            sequences. 
            Note: Each observation has d features, and there are a total of n_obs
            observation. 

        lengths: (l,) int ndarray 
            Specifies the length of each separate observation sequence in `data`
            There are l difference training sequences.

        n_states : int
            The number of hidden emitting states to use initially. 
        '''
        
        # Split the data into separate signals and pass to class
        self.data = data
        newstarts = np.cumsum(lengths)[:-1]
        self.signals = np.hsplit(data, newstarts) 
        self.trans = HMM._ltrtrans(n_states)
        self.trans, self.dists, newLL, iters = self._em(self.trans, self._ltrinit())

    def load(self, trans, dists):
        '''
        Initialize an HMM model using the provided data.

        Parameters
        ----------
        dists: (n_states,) list
            A list of Gaussian objects defining the emitting pdf's, one object for each 
            emitting state.

        trans : (n_states+1,n_states+1) ndarray
            The left-to-right transition probability table.  The rightmost column contains probability
            of transitioning to final state, and the last row the initial state's
            transition probabilities.   Note that all the rows need to add to 1. 
    
        '''

        self.trans, self.dists = trans, dists

    def _n_states(self):
        '''
        Get the number of emitting states used by the model.

        Return
        ------
        n_states : int
        The number of hidden emitting states to use initially. 
        '''

        return self.trans.shape[0]-1

    def _n_obs(self):
        '''
        Get the total number of observations in all signals in the data associated with the model.

        Return
        ------
        n_obs: int 
            The total number of observations in all the sequences combined.
        '''

        return self.data.shape[1]

    @staticmethod
    def _ltrtrans(n_states):
        '''
        Intialize the transition matrix (self.trans) with n_states emitting states (and an initial and 
        final non-emitting state) enforcing a left-to-right topology.  This means 
        broadly: no transitions from higher-numbered to lower-numbered states are 
        permitted, while all other transitions are permitted. 
        All legal transitions from a given state should be equally likely.

        The following exceptions apply:
        -The initial state may not transition to the final state
        -The final state may not transition (all transition probabilities from 
         this state should be 0)
    
        Parameter
        ---------
        n_states : int
            Number of emitting states for the transition matrix

        Return
        ------
        trans : (n_states+1,n_states+1) ndarray
            The left-to-right transition probability table initialized as described below.
        '''

        trans = np.zeros((n_states + 1, n_states + 1))
        trans[-1, :-1] = 1. / n_states
        for row in range(n_states):
            prob = 1./(n_states + 1 - row)
            for col in range(row, n_states+1):
                trans[row, col] = prob
        return trans

    def _ltrinit(self):
        '''
        Initial allocation of the observations to states in a left-to-right manner.
        It uses the observation data that is already available to the class.
    
        Note: Each signal consists of a number of observations. Each observation is 
        allocated to one of the n_states emitting states in a left-to-right manner
        by splitting the observations of each signal into approximately equally-sized 
        chunks of increasing state number, with the number of chunks determined by the 
        number of emitting states.
        If 'n' is the number of observations in signal, the allocation for signal is specified by:
        np.floor(np.linspace(0, n_states, n, endpoint=False))
    
        Returns
        ------
        states : (n_obs, n_states) ndarray
            Initial allocation of signal time-steps to states as a one-hot encoding.  Thus
            'states[:,j]' specifies the allocation of all the observations to state j.
        '''

        states = np.zeros((self._n_obs(), self._n_states()))
        i = 0
        for s in self.signals:
            vals = np.floor(np.linspace(0, self._n_states(), num=s.shape[1], endpoint=False))
            for v in vals:
                states[i][int(v)] = 1
                i += 1
        return np.array(states,dtype = bool)

    def viterbi(self, signal):
        '''
        See documentation for _viterbi()
        '''
        return HMM._viterbi(signal, self.trans, self.dists)

    @staticmethod
    def _viterbi(signal, trans, dists):
        '''
        Apply the Viterbi algorithm to the observations provided in 'signal'.
        Note: `signal` is a SINGLE observation sequence.
    
        Returns the maximum likelihood hidden state sequence as well as the
        log-likelihood of that sequence.

        Note that this function may behave strangely if the provided sequence
        is impossible under the model - e.g. if the transition model requires
        more observations than provided in the signal.
    
        Parameters
        ----------
        signal : (d,n) ndarray
            Signal for which the optimal state sequence is to be calculated.
            d is the dimension of each observation (number of features)
            n is the number of observations 
        
        trans : (n_states+1,n_states+1) ndarray
            The transition probability table.  The rightmost column contains probability
            of transitioning to final state, and the last row the initial state's
            transition probabilities.   Note that all the rows need to add to 1. 
    
        dists: (n_states,) list
            A list of Gaussian objects defining the emitting pdf's, one object for each 
            emitting  state.

        Return
        ------
        seq : (n,) ndarray
            The optimal state sequence for the signal (excluding non-emitting states)

        ll : float
            The log-likelihood associated with the sequence
        '''
        n_obs = signal.shape[1]
        n_states = trans.shape[0] - 1
        
        # Suppress log(0) warnings safely
        with np.errstate(divide='ignore'):
            log_trans = np.log(trans)
            
        V = np.zeros(n_states)
        
        # Store backpointers to reconstruct the optimal path later
        # backpointers[t, j] = the state at t-1 that maximized the probability of reaching state j at t
        backpointers = np.zeros((n_obs, n_states), dtype=int)
        
        # 1. Initialization (t = 0)
        for j in range(n_states):
            V[j] = dists[j].logf(signal[:, 0]) + log_trans[-1, j]
            
        # 2. Recursion (t = 1 to T - 1)
        for t in range(1, n_obs):
            V_new = np.zeros(n_states)
            for j in range(n_states):
                # Calculate the array of log probabilities arriving from all previous states i
                # This implements: V_{t-1}(i) + log(a_{i,j})
                probs = V + log_trans[:-1, j]
                
                # Find the max value and its index (the backpointer)
                best_prev_state = np.argmax(probs)
                backpointers[t, j] = best_prev_state
                
                # Update V_new with the max probability plus the current emission probability
                V_new[j] = dists[j].logf(signal[:, t]) + probs[best_prev_state]
                
            V = V_new # Update for the next time step
            
        # 3. Termination
        # Factor in the final transition to the terminal non-emitting state (index -1)
        final_probs = V + log_trans[:-1, -1]
        
        best_final_state = np.argmax(final_probs)
        max_ll = final_probs[best_final_state]
        
        # 4. Backtracking
        # Reconstruct the sequence from the end to the beginning
        best_path = np.zeros(n_obs, dtype=int)
        best_path[-1] = best_final_state
        
        # Loop backwards from T-2 down to 0
        for t in range(n_obs - 2, -1, -1):
            # The optimal state at time t is found by looking at the backpointer 
            # left by the optimal state at time t+1
            best_path[t] = backpointers[t + 1, best_path[t + 1]]
            
        return best_path, float(max_ll)
        # In this function, you may want to take log 0 and obtain -inf.
        # To avoid warnings about this, you can use np.seterr.

    def score(self, signal, seq):
        '''
        See documentation for _score()
        '''
        return HMM._score(signal, seq, self.trans, self.dists)

    @staticmethod
    def _score(signal, seq, trans, dists):
        '''
        Calculate the likelihood of an observation sequence and hidden state correspondence.
        Note: signal is a SINGLE observation sequence, and seq is the corresponding series of
        emitting states being scored.
    
        Returns the log-likelihood of the observation-states correspondence.

        Parameters
        ----------
        signal : (d,n) ndarray
            Signal for which the optimal state sequence is to be calculated.
            d is the dimension of each observation (number of features)
            n is the number of observations 
        
        seq : (n,) ndarray
            The state sequence provided for the signal (excluding non-emitting states)

        trans : (n_states+1,n_states+1) ndarray
            The transition probability table.  The rightmost column contains probability
            of transitioning to final state, and the last row the initial state's
            transition probabilities.   Note that all the rows need to add to 1. 
    
        dists: (n_states,) list
            A list of Gaussian objects defining the emitting pdf's, one object for each 
            emitting  state.

        Return
        ------
        ll : float
            The log-likelihood associated with the observation and state sequence under the model.
        '''
        # n is the total number of observations (T in your formula)
        n = signal.shape[1]
        ll = np.log(trans[-1,seq[0]])
        for t in range(n):
            current_state = seq[t]
            # for the probability part
            ll+= dists[current_state].logf(signal[:,t])

            if t< n-1:
                next_state = seq[t+1]
                ll+=np.log(trans[current_state,next_state])

        ll+= np.log(trans[seq[-1],-1])
        return ll

    def forward(self, signal):
        '''
        See documentation for _forward()
        '''
        return HMM._forward(signal, self.trans, self.dists)

    @staticmethod
    def _forward(signal, trans, dists):
        '''
        Apply the forward algorithm to the observations provided in 'signal' to
        calculate its likelihood.
        Note: `signal` is a SINGLE observation sequence.
    
        Returns the log-likelihood of the observation.

        Parameters
        ----------
        signal : (d,n) ndarray
            Signal for which the optimal state sequence is to be calculated.
            d is the dimension of each observation (number of features)
            n is the number of observations 
        
        trans : (n_states+1,n_states+1) ndarray
            The transition probability table.  The rightmost column contains probability
            of transitioning to final state, and the last row the initial state's
            transition probabilities.   Note that all the rows need to add to 1. 
    
        dists: (n_states,) list
            A list of Gaussian objects defining the emitting pdf's, one object for each 
            emitting  state.

        Return
        ------
        ll : float
            The log-likelihood associated with the observation under the model.
        '''
        # TODO: Implement this function
        n_obs = signal.shape[1]
        n_states = trans.shape[0] - 1
        
        # Helper function for the log-sum-exp trick to prevent underflow
        def log_sum_exp(z):
            z_max = np.max(z)
            if np.isneginf(z_max):
                return -np.inf
            return z_max + np.log(np.sum(np.exp(z - z_max)))
    
        # Suppress log(0) warnings temporarily as requested by the class docstring
        with np.errstate(divide='ignore'):
            log_trans = np.log(trans)
    
        # 1. Initialization (t = 0)
        L = np.zeros(n_states)
        for j in range(n_states):
            L[j] = dists[j].logf(signal[:, 0]) + log_trans[-1, j]
            
        # 2. Recursion (t = 1 to T - 1)
        for t in range(1, n_obs):
            L_new = np.zeros(n_states)
            for j in range(n_states):
                # Calculate the array of values inside the summation: log(a_ij) + L_{t-1}(i)
                # We slice [:-1, j] to ignore the initial state row
                z = log_trans[:-1, j] + L
                
                # Apply log-sum-exp trick and add emission probability
                log_sum = log_sum_exp(z)
                L_new[j] = dists[j].logf(signal[:, t]) + log_sum
                
            L = L_new # Update for the next time step
    
        # 3. Termination
        # Calculate array for final transitions to the terminal state (index -1)
        z_term = log_trans[:-1, -1] + L
        
        # Final marginalization using log-sum-exp
        ll = log_sum_exp(z_term)
        return ll

    def _calcstates(self, trans, dists):
        '''
        Calculate state sequences on the 'signals' maximizing the likelihood for 
        the given HMM parameters.
        
        Calculate the state sequences for each of the given 'signals', maximizing the 
        likelihood of the given parameters of a HMM model. This allocates each of the
        observations, in all the equences, to one of the states. 
    
        Use the state allocation to calculate an updated transition matrix.   
    
        IMPORTANT: As part of this updated transition matrix calculation, emitting states which 
        are not used in the new state allocation are removed. 
    
        In what follows, n_states is the number of emitting states described in trans, 
        while n_states' is the new number of emitting states.
        
        Note: signals consists of ALL the training sequences and is available
        through the class.
        
        Parameters
        ----------        
        trans : (n_states+1,n_states+1) ndarray
            The transition probability table.  The rightmost column contains probability
            of transitioning to final state, and the last row the initial state's
            transition probabilities.   Note that all the rows need to add to 1. 
    
        dists: (n_states,) list
            A list of Gaussian objects defining the emitting pdf's, one object for each 
            emitting  state.
    
        Return
        ------    
        states : bool (n_obs,n_states') ndarray
            The updated state allocations of each observation in all signals
        trans : (n_states'+ 1,n_states'+1) ndarray
            Updated transition matrix 
        ll : float
            Log-likelihood of all the data
        '''
        n_states = len(dists)
        n_obs_total = self._n_obs()
        
        # We will build the new state allocations here
        states = np.zeros((n_obs_total, n_states), dtype=bool)
        
        # We will count transitions to build the new transition matrix
        # Shape is (n_states + 1, n_states + 1) to account for initial and terminal states
        trans_counts = np.zeros((n_states + 1, n_states + 1))
        
        total_ll = 0.0
        current_obs_idx = 0
        
        # 1. Calculate state sequences for each signal using Viterbi[cite: 2]
        for signal in self.signals:
            seq, ll = HMM._viterbi(signal, trans, dists)
            total_ll += ll
            n = signal.shape[1]
            
            # Mark the active states in the boolean matrix
            for t in range(n):
                states[current_obs_idx + t, seq[t]] = True
                
            # Count the initial transition
            trans_counts[-1, seq[0]] += 1
            
            # Count the state-to-state transitions
            for t in range(n - 1):
                trans_counts[seq[t], seq[t+1]] += 1
                
            # Count the terminal transition
            trans_counts[seq[-1], -1] += 1
            
            current_obs_idx += n
            
        # 2. Identify and remove emitting states not used in the new state allocation[cite: 2]
        used_states_mask = np.sum(states, axis=0) > 0
        new_n_states = np.sum(used_states_mask)
        
        # Filter the states array
        states = states[:, used_states_mask]
        
        # Filter the transition counts to remove unused states
        # We need to keep the last row/col for the non-emitting states
        mask_with_non_emitting = np.append(used_states_mask, True)
        trans_counts = trans_counts[mask_with_non_emitting][:, mask_with_non_emitting]
        
        # 3. Calculate updated transition matrix
        new_trans = np.zeros((new_n_states + 1, new_n_states + 1))
        
        # Normalize rows to sum to 1
        for i in range(new_n_states + 1):
            row_sum = np.sum(trans_counts[i, :])
            if row_sum > 0:
                new_trans[i, :] = trans_counts[i, :] / row_sum
                
        return states, new_trans, total_ll
        # The core of this function involves applying the _viterbi function to each signal stored in the model.
        # Remember to remove emitting states not used in the new state allocation.

    def _updatecovs(self, states):
        '''
        Update estimates of the means and covariance matrices for each HMM state
    
        Estimate the covariance matrices for each of the n_states emitting HMM states for 
        the given allocation of the observations in self.data to states. 
        If self.diagcov is true, diagonal covariance matrices are returned.

        Parameters
        ----------
        states : bool (n_obs,n_states) ndarray
            Current state allocations for self.data in model
        
        Return
        ------
        covs: (n_states, d, d) ndarray
            The updated covariance matrices for each state

        means: (n_states, d) ndarray
            The updated means for each state
        '''
        n_obs, n_states = states.shape
        d = self.data.shape[0]
        
        covs = np.zeros((n_states, d, d))
        means = np.zeros((n_states, d))
        
        for j in range(n_states):
            # Extract the observations currently assigned to state j
            state_data = self.data[:, states[:, j]]
            N_j = state_data.shape[1]
            
            if N_j == 0:
                # If a class has no observations, assign mean of zero and identity covariance
                means[j] = np.zeros(d)
                covs[j] = np.eye(d)
            else:
                # Calculate mean
                means[j] = np.mean(state_data, axis=1)
                
                # Calculate covariance
                # np.cov treats each row as a variable and columns as observations
                # ddof=0 gives the maximum likelihood estimate (divide by N, not N-1)
                # Calculate covariance
                # np.cov treats each row as a variable and columns as observations
                # ddof=0 gives the maximum likelihood estimate (divide by N, not N-1)
                full_cov = np.atleast_2d(np.cov(state_data, ddof=0))
                
                # Handle the case where the calculated covariance is a zero matrix[cite: 2]
                if np.all(full_cov == 0):
                    full_cov = np.eye(d)
                    
                # Discard non-diagonal elements if diagcov is required[cite: 2]
                if self.diagcov:
                    covs[j] = np.diag(np.diag(full_cov))
                else:
                    covs[j] = full_cov
                    
        return covs, means
        pass
        # In this method, if a class has no observations, assign it a mean of zero
        # In this method, estimate a full covariance matrix and discard the non-diagonal elements
        # if a diagonal covariance matrix is required.
        # In this method, if a zero covariance matrix is obtained, assign an identity covariance matrix
               
    def _em(self, trans, states):
        '''
        Perform parameter estimation for a hidden Markov model (HMM).
    
        Perform parameter estimation for an HMM using multi-dimensional Gaussian 
        states.  The training observation sequences, signals,  are available 
        to the class, and states designates the initial allocation of emitting states to the
        signal time steps.   The HMM parameters are estimated using Viterbi 
        re-estimation. 
        
        Note: It is possible that some states are never allocated any 
        observations.  Those states are then removed from the states table, effectively redusing
        the number of emitting states. In what follows, n_states is the original 
        number of emitting states, while n_states' is the final number of 
        emitting states, after those states to which no observations were assigned,
        have been removed.
    
        Parameters
        ----------
        trans : (n_states+1,n_states+1) ndarray
            The left-to-right transition probability table.  The rightmost column contains probability
            of transitioning to final state, and the last row the initial state's
            transition probabilities.   Note that all the rows need to add to 1. 
        
        states : (n_obs, n_states) ndarray
            Initial allocation of signal time-steps to states as a one-hot encoding.  Thus
            'states[:,j]' specifies the allocation of all the observations to state j.
        
        Return
        ------
        trans : (n_states'+1,n_states'+1) ndarray
            Updated transition probability table

        dists : (n_states',) list
            Gaussian object of each component.

        newLL : float
            Log-likelihood of parameters at convergence.

        iters: int
            The number of iterations needed for convergence
        '''

        covs, means = self._updatecovs(states) 
        dists = [Gaussian(mean=means[i], cov=covs[i]) for i in range(len(covs))]
        
        oldstates, trans, oldLL = self._calcstates(trans, dists)
        
        converged = False
        iters = 0
        newLL = oldLL
        
        while not converged and iters < self.maxiters:
            # Update parameters based on current state allocations[cite: 2]
            covs, means = self._updatecovs(oldstates)
            dists = [Gaussian(mean=means[i], cov=covs[i]) for i in range(len(covs))]
            
            # Recalculate optimal state sequences and transition matrix[cite: 2]
            oldstates, trans, newLL = self._calcstates(trans, dists)
            
            # Test for convergence using relative error tolerance[cite: 2]
            if oldLL != 0:
                rel_error = abs((newLL - oldLL) / oldLL)
            else:
                rel_error = float('inf')
                
            if rel_error < self.rtol:
                converged = True
                
            oldLL = newLL
            iters += 1
            
        if iters >= self.maxiters:
            from warnings import warn
            warn("Maximum number of iterations reached - HMM parameters may not have converged")[cite: 2]
            
        return trans, dists, newLL, iters
        
    def sample(self):
        '''
        Draw samples from the HMM using the present model parameters. The sequence
        terminates when the final non-emitting state is entered. For the
        left-to-right topology used, this should happen after a finite number of 
        samples is generated, modeling a finite observation sequence. 
        
        Returns
        -------
        samples: (n,) ndarray
            The samples generated by the model
        states: (n,) ndarray
            The state allocation of each sample. Only the emitting states are 
            recorded. The states are numbered from 0 to n_states-1.

        Sample usage
        ------------
        Example below commented out, since results are random and thus not suitable for doctesting.
        However, the example is based on the model fit in the doctests for the class.
        #>>> samples, states = hmm.samples()
        #>>> print(samples)
        #[ 0.9515792   0.9832767   1.04633007  1.01464327  0.98207072  1.01116689
        #  0.31622856  0.20819263  3.57707616]           
        #>>> print(states)   #These will differ for each call
        #[1 1 1 1 1 1 2 2 3]
        '''
        
        #######################################################################
        import scipy.interpolate as interpolate
        def draw_discrete_sample(discr_prob):
            '''
            Draw a single discrete sample from a probability distribution.
            
            Parameters
            ----------
            discr_prob: (n,) ndarray
                The probability distribution.
                Note: sum(discr_prob) = 1
                
            Returns
            -------
            sample: int
                The discrete sample.
                Note: sample takes on the values in the set {0,1,n-1}, where
                n is the the number of discrete probabilities.
            '''

            if not np.sum(discr_prob) == 1:
                raise ValueError('The sum of the discrete probabilities should add to 1')
            x = np.cumsum(discr_prob)
            x = np.hstack((0.,x))
            y = np.array(range(len(x)))
            fn = interpolate.interp1d(x,y)           
            r = np.random.rand(1)
            return np.array(np.floor(fn(r)),dtype=int)[0]
        #######################################################################
        n_states = self._n_states()
        samples_list = []
        states_list = []
        
        # 1. Initial transition: Draw the first emitting state using the 
        # transition probabilities from the initial non-emitting state (last row)
        current_state = draw_discrete_sample(self.trans[-1, :])
        
        # 2. Loop until the terminal non-emitting state is reached. 
        # The terminal state is the final column (index n_states)
        while current_state != n_states:
            # Record the current emitting state
            states_list.append(current_state)
            
            # Draw an observation sample from the current state's Gaussian density
            # .sample(n=1) returns shape (d, 1), so we flatten it to (d,) for list appending
            obs = self.dists[current_state].sample(n=1).flatten()
            samples_list.append(obs)
            
            # Transition to the next state using the current state's transition row
            current_state = draw_discrete_sample(self.trans[current_state, :])
            
        # 3. Formatting the output
        # states must be an (n,) array[cite: 2]
        states = np.array(states_list, dtype=int)
        
        # samples_list is a list of (d,) arrays.
        # np.array(samples_list) creates an (n, d) array.
        # Transposing it yields the required (d, n) ndarray format[cite: 2]
        samples = np.array(samples_list).T
        
        return samples, states
     # Using the function defined above, draw samples from the HMM 

if __name__ == "__main__":
    import doctest
    doctest.testmod() 
