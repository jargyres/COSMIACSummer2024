import numpy as np
import matplotlib.pyplot as plt

# Define the parameters
carrier_frequency_GHz = 2.4  # Carrier frequency in GHz
sampling_rate_Msps = 20     # Sampling rate in Msamples per second
num_samples = 1024          # Number of samples to generate

# Convert frequencies to Hz
carrier_frequency_Hz = carrier_frequency_GHz * 1e9
sampling_rate_Hz = sampling_rate_Msps * 1e6

# Calculate the time period based on the carrier frequency
period = 1 / carrier_frequency_Hz

# Generate the time array for the full signal
full_time = np.arange(0, num_samples) / sampling_rate_Hz

# Generate the carrier wave for the full time array
carrier_wave_full = np.sin(2 * np.pi * carrier_frequency_Hz * full_time)

# Extract 1024 samples starting from the beginning
# carrier_wave_1024 = carrier_wave_full[:num_samples]

# Plot the extracted 1024 samples
plt.figure(figsize=(10, 6))
plt.plot(carrier_wave_full)
plt.title('Extracted 1024 Samples of Carrier Wave at 2.4 GHz, Sampled at 20 Msps')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()