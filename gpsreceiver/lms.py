import numpy as np

class LMS:
    """ 
    Implements the Complex LMS algorithm for COMPLEX valued data.
        (Algorithm 3.2 - book: Adaptive Filtering: Algorithms and Practical
                                                       Implementation, Diniz)

    Base class for other LMS-based classes

    ...

    Attributes
    ----------    
    . step: (float)
        Convergence (relaxation) factor.
    . filter_order : (int)
        Order of the FIR filter.
    . init_coef : (row np.array)
        Initial filter coefficients.  (optional)
    . d: (row np.array)
        Desired signal. 
    . x: (row np.array)
        Signal fed into the adaptive filter. 
    
    Methods
    -------
    fit(d, x)
        Fits the coefficients according to desired and input signals
    
    predict(x)
        After fitted, predicts new outputs according to new input signal    
    """
    def __init__(self, step, filter_order, init_coef = None):        
        self.step = step
        self.filter_order = filter_order
        self.init_coef = np.array(init_coef)
    
        # Initialization Procedure
        self.n_coef = self.filter_order + 1
        self.d = None
        self.x = None
        self.n_iterations = None
        self.error_vector = None
        self.output_vector = None
        self.coef_vector = None
        
    def __str__(self):
        """ String formatter for the class"""
        return "LMS(step={}, filter_order={})".format(self.step, self.filter_order)
        
    def fit(self, d, x):
        """ Fits the LMS coefficients according to desired and input signals
        
        Arguments:
            d {np.array} -- desired signal
            x {np.array} -- input signal
        
        Returns:
            {np.array, np.array, np.array} -- output_vector, error_vector, coef_vector
        """
        # Pre allocations
        self.d = np.array(d)        
        self.x = np.array(x)        
        self.n_iterations = len(self.d)
        self.output_vector = np.zeros([self.n_iterations], dtype=complex)
        self.error_vector = np.zeros([self.n_iterations], dtype=complex)
        self.coef_vector = np.zeros([self.n_iterations+1, self.n_coef], dtype=complex)

        # Initial State Weight Vector if passed as argument        
        self.coef_vector[0] = np.array([self.init_coef]) if self.init_coef is not None else self.coef_vector[0]

        # Improve source code regularity
        prefixed_x = np.append(np.zeros([self.n_coef-1]), self.x)        
        X_tapped = self.rolling_window(prefixed_x, self.n_coef)

        for k in np.arange(self.n_iterations):                    
            regressor = X_tapped[k]
            # print("X_Tapped[{}]={}".format(k, regressor))

            self.output_vector[k] = np.dot(np.conj(self.coef_vector[k]), regressor)            
            self.error_vector[k] = self.d[k]-self.output_vector[k]
            self.coef_vector[k+1] = self.coef_vector[k]+self.step*np.conj(self.error_vector[k])*regressor
                        
        return self.output_vector, self.error_vector, self.coef_vector
    
    def predict(self, x):
        """ Makes predictions for a new signal after weights are fit
        
        Arguments:
            x {row np.array} -- new signal
        
        Returns:
            float -- resulting output""" 

        # taking the last n_coef iterations of x and making w^t.x

        # return np.dot(self.coef_vector[-1], x[:-self.n_coef])
        return np.dot(self.coef_vector, x)

        
    
    def rolling_window(self, x, window):
        """ Creates a N-sized rolling window for vector x
        
        Arguments:
            x {np.array} -- input vector
            window {int} -- window size
        
        Returns:
            [np.array] -- array of x window's
        """
        shape = x.shape[:-1] + (x.shape[-1] - window + 1, window)
        strides = x.strides + (x.strides[-1],)
        return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
