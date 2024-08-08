import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from typing import Callable
'''
Creates a ULA antenna array
'''
class AntennaArray:
    def __init__(self, num_elements: int, wavelength_spacing: float) -> None:
        self.wavelength_spacing = wavelength_spacing
        self.num_elements = num_elements

    def generate_signal(self, sample_rate: int, num_samples: int, frequency: int):
        t = np.arange(num_samples)/sample_rate
        signal = np.exp(2j*np.pi*frequency*t)
        return signal
    
    def apply_AWGN(self, signal: npt.NDArray[np.complex128], variance: float):
        #We want to apply the noise after the steering vector is applied, 
        #because each element experiences an independent noise signal 
        #(we can do this because AWGN with a phase shift applied is still AWGN)
        N = len(signal[0])
        n = np.random.randn(self.num_elements, N) + 1j*np.random.randn(self.num_elements, N)
        signal_with_noise = signal + variance*n
        return signal_with_noise
    
    def simulate_RX(self, samplerate: int, num_samples: int, frequency: int, direction_of_arrival:float):
        signal = self.generate_signal(sample_rate=samplerate, num_samples=num_samples, frequency=frequency)
        received_signal = self.receivedSignal(tx=signal, theta=direction_of_arrival)
        return received_signal
    
    def simulate_RX_with_noise(self, samplerate: int, 
                               num_samples: int, 
                               frequency: int, 
                               direction_of_arrival:float, 
                               noise_variance:float) -> npt.NDArray[np.complex128]:
        r = self.simulate_RX(samplerate=samplerate, num_samples=num_samples, frequency=frequency, direction_of_arrival=direction_of_arrival)
        return self.apply_AWGN(signal=r, variance=noise_variance)
    
    def simulate_multiple_tx(self, 
                             num_samples: int, 
                             sample_rate: int, 
                             thetas: npt.NDArray[np.float_], 
                             signal_powers_theta: npt.NDArray[np.float_], 
                             signal_frequencies_theta: npt.NDArray[np.int_], 
                             AWGN_variance: float) -> npt.NDArray[np.complex128]:

        theta_radians = thetas / 180 * np.pi
        t = np.arange(num_samples)/sample_rate

        final_signals = np.zeros((len(thetas), self.num_elements, num_samples), dtype=np.complex128)
        for k in range(len(theta_radians)):
            s = np.exp(-2j * np.pi * self.wavelength_spacing * np.arange(self.num_elements) * np.sin(theta_radians[k])).reshape(-1,1)
            tone = np.exp(2j*np.pi*signal_frequencies_theta[k]*t).reshape(1,-1)
            final_signals[k] = signal_powers_theta[k] * s @ tone

        
        return self.apply_AWGN(np.sum(final_signals, axis=0), AWGN_variance)
        
    '''
    Returns the steering vectors for this ULA in direction theta
    Theta given as angle, not radians
    '''
    def steeringVectors(self, theta: float) -> npt.NDArray[np.complex128]:
        thetas = theta / 180 * np.pi # convert to radians
        s = np.exp(-2j * np.pi * self.wavelength_spacing * np.arange(self.num_elements) * np.sin(thetas)) # steering vector
        return s
    
    def receivedSignal(self, tx: npt.NDArray[np.complex128], theta: float) -> npt.NDArray[np.complex128]:
        steeringVectors = self.steeringVectors(theta=theta)
        print(steeringVectors)
        # we have to do a matrix multiplication of s and tx, which currently are both 1D, so we have to make them 2D with reshape
        steeringVectors = steeringVectors.reshape(-1,1) #num_elements x 1
        tx = tx.reshape(1,-1) #1 x num_samples
        assert len(steeringVectors[0]) == len(tx)
        r = steeringVectors @ tx # matrix  multiply r = num_elements x num_samples.  r is now going to be a 2D array, 1d is time and 1d is spatial
        return r
    
    '''
    Gives the total signal power at a certain angle theta
    '''
    def doa_response_theta(self, tx: npt.NDArray[np.complex128], theta: float) -> float:
        w = np.exp(-2j * np.pi * self.wavelength_spacing * np.arange(self.num_elements) * np.sin(theta)) # Conventional, aka delay-and-sum, beamformer
        X_weighted = w.conj().T @ tx # apply our weights.
        total_signal_power = 10*np.log10(np.var(X_weighted)) # power in signal, in dB so its easier to see small and large lobes at the same time
        return total_signal_power
    
    '''
    Gives the total signal power at a certain angle theta for a certain beamforming weight calculator function
    '''
    def doa_response_theta_custom_weights(self, tx: npt.NDArray[np.complex128], theta: float, weight_function) -> float:
        w = weight_function(tx, theta)
        X_weighted = w.conj().T @ tx # apply our weights.
        total_signal_power = 10*np.log10(np.var(X_weighted)) # power in signal, in dB so its easier to see small and large lobes at the same time
        return total_signal_power
    
    '''
    DOA results; they correspond to the received power at each angle after applying the beamformer.
    '''
    def total_doa_response(self, tx: npt.NDArray[np.complex128], plot: bool):

        theta_scan = np.linspace(-1*np.pi, np.pi, 1000) # 1000 different thetas between -180 and +180 degrees
        results = []
        for theta_i in theta_scan:
            
            results.append(self.doa_response_theta(tx=tx, theta=theta_i))

        results -= np.max(results) # normalize

        results = np.asanyarray(results)

        if(not plot):
            return theta_scan, results
        else:
            # print angle that gave us the max value
            print(theta_scan[np.argmax(results)] * 180 / np.pi)

            fig, (ax1) = plt.subplots(1, 1, figsize=(7, 3))
            ax1.plot(theta_scan*180/np.pi, results) # lets plot angle in degrees
            ax1.set_xlabel("Theta [Degrees]")
            ax1.set_ylabel("DOA Metric")
            ax1.set_title("Total Signal Power vs Angle")
            ax1.grid()
            plt.show()
            #fig.savefig('../_images/doa_conventional_beamformer.svg', bbox_inches='tight')

            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
            ax.plot(theta_scan, results) # MAKE SURE TO USE RADIAN FOR POLAR
            ax.set_theta_zero_location('N') # make 0 degrees point up
            ax.set_theta_direction(-1) # increase clockwise
            ax.set_rlabel_position(55)  # Move grid labels away from other labels
            ax.set_title("Polar Plot of DOA Metric")
            plt.show()
    '''
    DOA results; they correspond to the received power at each angle after applying the beamformer.
    '''
    def total_doa_response_custom_weights(self, tx: npt.NDArray[np.complex128], plot: bool, weight_function):

        theta_scan = np.linspace(-1*np.pi, np.pi, 1000) # 1000 different thetas between -180 and +180 degrees

        results = []

        for theta_i in theta_scan:
            
            # results.append(self.doa_response_theta(tx=tx, theta=theta_i))
            results.append(self.doa_response_theta_custom_weights(tx=tx, theta=theta_i, weight_function=weight_function))


        results -= np.max(results) # normalize

        results = np.asanyarray(results)

        if(not plot):
            return theta_scan, results
        else:
            # print angle that gave us the max value
            print(theta_scan[np.argmax(results)] * 180 / np.pi)

            fig, (ax1) = plt.subplots(1, 1, figsize=(7, 3))
            ax1.plot(theta_scan*180/np.pi, results) # lets plot angle in degrees
            ax1.set_xlabel("Theta [Degrees]")
            ax1.set_ylabel("DOA Metric")
            ax1.set_title("Total Signal Power vs Angle")
            ax1.grid()
            plt.show()
            #fig.savefig('../_images/doa_conventional_beamformer.svg', bbox_inches='tight')

            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
            ax.plot(theta_scan, results) # MAKE SURE TO USE RADIAN FOR POLAR
            ax.set_theta_zero_location('N') # make 0 degrees point up
            ax.set_theta_direction(-1) # increase clockwise
            ax.set_rlabel_position(55)  # Move grid labels away from other labels
            ax.set_title("Polar Plot of DOA Metric")
            plt.show()

    '''
    calculate and plot the quiescent antenna pattern (array response)
      when steered towards a certain direction, 
      which will tell us the arrays natural response if we don't do any additional beamforming.
    '''
    def beam_pattern(self, theta: float):
        N_fft = 512
        theta_degrees = theta
        # theta_degrees = 20 # there is no SOI, we arent processing samples, this is just the direction we want to point at
        thetas = theta_degrees / 180 * np.pi
        w = np.exp(-2j * np.pi * self.wavelength_spacing * np.arange(self.num_elements) * np.sin(thetas)) # conventional beamformer
        w = np.conj(w) # or else our answer will be negative/inverted
        w_padded = np.concatenate((w, np.zeros(N_fft - self.num_elements))) # zero pad to N_fft elements to get more resolution in the FFT
        w_fft_dB = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(w_padded)))**2) # magnitude of fft in dB
        w_fft_dB -= np.max(w_fft_dB) # normalize to 0 dB at peak

        # Map the FFT bins to angles in radians
        theta_bins = np.arcsin(np.linspace(-1, 1, N_fft)) # in radians

        # find max so we can add it to plot
        theta_max = theta_bins[np.argmax(w_fft_dB)]

        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        ax.plot(theta_bins, w_fft_dB) # MAKE SURE TO USE RADIAN FOR POLAR
        ax.plot([theta_max], [np.max(w_fft_dB)],'ro')
        ax.text(theta_max - 0.1, np.max(w_fft_dB) - 4, np.round(theta_max * 180 / np.pi))
        ax.set_theta_zero_location('N') # make 0 degrees point up
        ax.set_theta_direction(-1) # increase clockwise
        ax.set_rlabel_position(55)  # Move grid labels away from other labels
        ax.set_thetamin(-90) # only show top half
        ax.set_thetamax(90)
        ax.set_ylim([-30, 1]) # because there's no noise, only go down 30 dB
        ax.set_title("ULA Beam Pattern at {} degrees".format(theta))
        plt.show()

        

    
# ula = AntennaArray(4, 0.5)

# sample_rate = 1e6
# N = 5000000 # number of samples to simulate

# # Create a tone to act as the transmitted signal
# t = np.arange(N)/sample_rate
# f_tone = 0.02e6
# tx = np.exp(2j*np.pi*f_tone*t)


# r = ula.receivedSignal(tx, 45)
# r = ula.simulate_RX_with_noise(samplerate=sample_rate, num_samples=N, frequency=f_tone, direction_of_arrival=45, noise_variance=0.5)


# # #We want to apply the noise after the steering vector is applied, 
# # #because each element experiences an independent noise signal 
# # #(we can do this because AWGN with a phase shift applied is still AWGN)
# # n = np.random.randn(3, N) + 1j*np.random.randn(3, N)
# # r = r + 0.5*n

# # ula.total_doa_response(tx=r, plot=True)

# ula.beam_pattern(0)
