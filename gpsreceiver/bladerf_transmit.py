from bladerf import _bladerf
import numpy as np
from utils import Utils
import matplotlib.pyplot as plt
import time

sdr = _bladerf.BladeRF(device_identifier="*:serial=104")
print("opening")
# sdr = _bladerf.BladeRF()
print("done opening")

tx_ch = sdr.Channel(_bladerf.CHANNEL_TX(0)) # give it a 0 or 1
utils = Utils()
sample_rate = int(15e6)
center_freq = int(2.4e9)
gain = 60 # -15 to 60 dB. for transmitting, start low and slowly increase, and make sure antenna is connected
num_samples = 8192
# repeat = 30 # number of times to repeat our signal
# print('duration of transmission:', num_samples/sample_rate*repeat, 'seconds')
u = Utils()
start_binary = np.array([1,1,1,1,1,1,1,0])
real_data = np.random.randint(2, size=8)
input_arr = np.append(start_binary,real_data)
# input_arr = np.ones(16)
input_arr = np.array([1,1,0,0, 1,1,1,1, 1,1,0,0, 1,1,1,1])
# input_arr = np.array([1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1])


# print(input_arr)


# utils.gen_bpsk_carrier_signal(sample_rate, center_freq, num_samples)
# buf = utils.bpsk_modulate(input_arr)

# print(buf)

# plt.plot(buf)
# plt.show()

# input_arr += 1

# samples_per_symbol = 256

# phase_signal = np.repeat(input_arr, samples_per_symbol)

samples_per_symbol = 512


phase_sequence = np.array([2*x - 1 for x in input_arr])
# phase_sequence = np.array([x+1 for x in input_arr])

# Generate the continuous phase signal with the desired samples per symbol
phase_signal = np.repeat(phase_sequence, samples_per_symbol)
# print(phase_signal)
i = np.zeros(num_samples)
q = np.zeros(num_samples)
for k in range(num_samples):
    i[k] = np.round(phase_signal[k] * 2048.0 * np.cos(2.0 * k * np.pi / num_samples))
    q[k] = np.round(phase_signal[k] * 2048.0 * np.sin(2.0 * k * np.pi / num_samples))
    if(i[k] > 2047):
        i[k] = 2047
    if(i[k] < -2048):
        i[k] = -2048
    if(q[k] > 2047):
        q[k] = 2047
    if(i[k] < -2048):
        q[k] = -2048          

carrier_signal = i + 1j * q

# buf = phase_signal * carrier_signal
# print(input_arr)
# input_arr = np.array([1,1, 0, 0, 1, 1, 0,0])

# x = np.zeros(100) #leading floor

# x = np.append(x,(input_arr+1)*np.exp(1j*2*np.pi*np.random.rand()))
# x = np.append(x,np.zeros(int(np.random.uniform(0.5,1)*n))) #trailing floor
# x = np.append(x,np.zeros(int(100)))


# buf = x

# # buf = u.bpsk_modulate(input_arr, int(20e6), int(5e6))
# buf = u.gen_bpsk_signal(input_arr, int(20e6))
# buf = u.am_modulation(input_arr)

# plt.plot(buf)
# plt.show()


# # Generate IQ samples to transmit (in this case, a simple tone)
t = np.arange(num_samples) / sample_rate
f_tone = 1e6
samples = np.exp(1j * 2 * np.pi * f_tone * t) # will be -1 to +1


# plt.plot(samples)
# plt.show()
# print(samples)

# samples_per_symbol = 256


# phase_sequence = np.array([2*x - 1 for x in input_arr])

# # Generate the continuous phase signal with the desired samples per symbol
# phase_signal = np.repeat(phase_sequence, samples_per_symbol)
# print(phase_signal)
# samples = phase_signal * samples

# plt.plot(phase_signal)
# plt.show()


# samples = np.ones(num_samples)
# samples = (np.random.rand(num_samples) + 1j*np.random.rand(num_samples)) * phase_signal

# plt.plot(samples)
# plt.show()

Fs = (2.4e9 * 4)# sample rate
Ts = 1/Fs # sample period
N = num_samples # number of samples to simulate

t = Ts*np.arange(N)

samples = np.exp(1j*2*np.pi*(2.4e9)*t)

# samples = phase_signal * samples
# samples[len(samples)//2:] = 0
# plt.plot(samples)
# plt.show()
# samples = carrier_signal
# print(input_arr)
samples = phase_signal / 2
# samples[len(samples)//2:] = 0
# plt.plot(samples.real)
# plt.show()

samples = samples.astype(np.complex64)


# samples *= 32767 # scale so they can be stored as int16s
samples *= 2048 # scale so they can be stored as int16s

# samples = phase_signal * samples

samples = samples.view(np.int16)
buf = samples.tobytes() # convert our samples to bytes and use them as transmit buffer



# hshshs = np.zeros(num_samples)

# buf = (phase_signal) + 1j * hshshs

# plt.plot(buf)
# plt.show()

# buf = buf.astype(np.complex64)

# buf *= 32767

# buf = buf.view(np.int16)

# buf = buf.tobytes()

# print(buf)







tx_ch.frequency = center_freq
tx_ch.sample_rate = sample_rate
tx_ch.bandwidth = sample_rate/2
tx_ch.gain = gain

# Setup synchronous stream
sdr.sync_config(layout=_bladerf.ChannelLayout.TX_X1, # or TX_X2
                fmt=_bladerf.Format.SC16_Q11, # int16s
                num_buffers=16,
                buffer_size=8192,
                num_transfers=8,
                stream_timeout=3500)

print("Starting transmit!")
tx_ch.enable = True
while True:
    # time.sleep(0.5)
    try:
        sdr.sync_tx(buf, num_samples) # write to bladeRF
    except KeyboardInterrupt:
        break


print("Stopping transmit")
tx_ch.enable = False
print("Closing Device")
sdr.close()