import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view

def psk_demodulation(signal, samples_per_symbol):
    # Generate the carrier signal    
    t = np.arange(0,len(signal))/10e3
    fc = 0.5e3
    carrier_signal = np.sin(2*np.pi*fc*t)
    
    # Multiply rx signal by carrier
    x = carrier_signal*signal
    
    # Integrate an sample
    x = np.sum(sliding_window_view(x, window_shape=samples_per_symbol), axis=1)    
    symbols = x[::20]
    
    # Get bits from symbols
    bits = symbols<0
    
    return bits.astype(int)


def psk_modulation(bits, samples_per_symbol):
    modulation_order = 2
    # Define the phase states for the chosen modulation order
    phase_states = np.linspace(0, 2*np.pi, modulation_order, endpoint=False)

    # Map the bits to the corresponding phase states
    phase_sequence = [phase_states[int(b)] for b in bits]

    # Generate the continuous phase signal with the desired samples per symbol
    phase_signal = np.repeat(phase_sequence, samples_per_symbol)

    # Generate the carrier signal
    t = np.arange(0,len(phase_signal))/10e3
    fc = 0.5e3
    carrier_signal = np.sin(2*np.pi*fc*t+phase_signal)

    plt.plot(carrier_signal)
    plt.show()

    return carrier_signal


# # Example inputs
# bits = [0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1]  # Sequence of bits
# samples_per_symbol = 64  # Number of samples per symbol

# # Compute the PSK signal
# psk_signal = psk_modulation(bits, samples_per_symbol)

# print(psk_signal.shape)

# demod_data = psk_demodulation(psk_signal, samples_per_symbol)

# print(demod_data)

