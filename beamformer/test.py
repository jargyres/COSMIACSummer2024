from core.array import AntennaArray
from core.mvdr import MVDR_Beamformer
import numpy as np
import matplotlib.pyplot as plt

ula = AntennaArray(4, 0.5)

mvdr = MVDR_Beamformer(array=ula)



sample_rate = 1e6
f_carrier = 0.02e6
N = 1024 # number of samples to simulate

thetas = np.array([20, 25, -40])
signal_powers_theta = np.array([1.0, 1.0, 0.1])
signal_frequencies_theta = np.array([0.01e6, 0.02e6, 0.03e6])

X = ula.simulate_multiple_tx(num_samples=N, 
                         sample_rate=sample_rate, 
                         thetas=thetas, 
                         signal_powers_theta=signal_powers_theta, 
                         signal_frequencies_theta=signal_frequencies_theta,
                         AWGN_variance=0.001)


ula.total_doa_response_custom_weights(X, True, mvdr.w_mvdr)

