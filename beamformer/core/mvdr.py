from core.array import AntennaArray
import numpy as np

ula = AntennaArray(4, 0.5)

class MVDR_Beamformer:
    def __init__(self, array: AntennaArray) -> None:
            self.array = array
    # theta is the direction of interest, in radians, and X is our received signal
    def w_mvdr(self, X, theta):
        
        s = np.exp(-2j * np.pi * self.array.wavelength_spacing * np.arange(self.array.num_elements) * np.sin(theta)) # steering vector in the desired direction theta
        s = s.reshape(-1,1) # make into a column vector (size 3x1)
        R = (X @ X.conj().T)/X.shape[1] # Calc covariance matrix. gives a Nr x Nr covariance matrix of the samples
        Rinv = np.linalg.pinv(R) # 3x3. pseudo-inverse tends to work better/faster than a true inverse
        w = (Rinv @ s)/(s.conj().T @ Rinv @ s) # MVDR/Capon equation! numerator is 3x3 * 3x1, denominator is 1x3 * 3x3 * 3x1, resulting in a 3x1 weights vector
        return w
    
    def power_mvdr(self, theta, X):
        s = np.exp(-2j * np.pi * self.array.wavelength_spacing * np.arange(self.array.num_elements) * np.sin(theta)) # steering vector in the desired direction theta
        s = s.reshape(-1,1) # make into a column vector (size 3x1)
        R = (X @ X.conj().T)/X.shape[1] # Calc covariance matrix. gives a Nr x Nr covariance matrix of the samples
        Rinv = np.linalg.pinv(R) # 3x3. pseudo-inverse tends to work better/faster than a true inverse
        return 1/(s.conj().T @ Rinv @ s).squeeze()