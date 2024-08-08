from bladerf import _bladerf
import numpy as np
from utils import Utils
import matplotlib.pyplot as plt
import time

print("opening")
sdr = _bladerf.BladeRF(device_identifier="*:serial=35d")
print("done opening")


print("Loopback")
sdr.set_loopback(1)
print(sdr.get_loopback())



tx_ch = sdr.Channel(_bladerf.CHANNEL_TX(0)) # give it a 0 or 1
rx_ch = sdr.Channel(_bladerf.CHANNEL_RX(0))
utils = Utils()
sample_rate = int(10e6)
tx_center_freq = int(1e9)
# tx_center_freq = int(75e6)

rx_freq_offset = int(1e6)
rx_center_freq = tx_center_freq - rx_freq_offset
gain_tx = 60 # -15 to 60 dB. for transmitting, start low and slowly increase, and make sure antenna is connected
gain_rx = 60 # -15 to 60 dB. for transmitting, start low and slowly increase, and make sure antenna is connected

num_samples = 4096

u = Utils()
start_binary = np.array([1,1,1,1,1,1,1,0])
real_data = np.random.randint(2, size=8)
input_arr = np.append(start_binary,real_data)
# input_arr = np.ones(16)
input_arr = np.array([1,1,0,0, 0,0,1,1, 1,1,0,0, 1,1,1,1])

# i = np.zeros(num_samples)
# q = np.zeros(num_samples)
# for k in range(num_samples):
#     i[k] = np.round(2048.0 * np.cos(2.0 * k * np.pi / num_samples))
#     q[k] = np.round(2048.0 * np.sin(2.0 * k * np.pi / num_samples))
#     if(i[k] > 2047):
#         i[k] = 2047
#     if(i[k] < -2048):
#         i[k] = -2048
#     if(q[k] > 2047):
#         q[k] = 2047
#     if(i[k] < -2048):
#         q[k] = -2048          

# carrier_signal = i + 1j * q


# fs = sample_rate * 4

# fc = tx_center_freq

# ts = 1 / float(fs)

# t = np.arange(0, num_samples * ts, ts)

# i = np.cos(2 * np.pi * t * fc) * 2 ** 14

# q = np.sin(2 * np.pi * t * fc) * 2 ** 14

# bpsk_carrier_signal = i + 1j * q

Fs = (sample_rate)# sample rate
Ts = 1/Fs # sample period
N = num_samples # number of samples to simulate

t = Ts*np.arange(N)

bpsk_carrier_signal = np.exp(1j*2*np.pi*(1e5)*t)

# plt.plot(bpsk_carrier_signal)
# plt.show()



samples_per_symbol = 256
phase_sequence = np.array([2*x - 1 for x in input_arr])
# phase_sequence = np.array(input_arr)

# # Generate the continuous phase signal with the desired samples per symbol
phase_signal = np.repeat(phase_sequence, samples_per_symbol)


# # samples = carrier_signal
# # samples = bpsk_carrier_signal * phase_signal * 2047


# x_degrees = phase_sequence*360/2.0
# x_radians = x_degrees*np.pi/180
# x_symbols = np.cos(x_radians) + 0.0j*np.sin(x_radians) #create complex samples
# samples = np.repeat(x_symbols, samples_per_symbol)
# samples *= 2048

# print(samples)

# plt.plot(samples)
# plt.show()
samples = phase_signal * 32767

samples = samples.astype(np.complex64)


# samples *= 32767 # scale so they can be stored as int16s



# samples = phase_signal * samples
# print(samples)
# plt.plot(samples)
# plt.show()

samples = samples.view(np.int16)
buf = samples.tobytes() # convert our samples to bytes and use them as transmit buffer


tx_ch.frequency = tx_center_freq
tx_ch.sample_rate = sample_rate
tx_ch.bandwidth = sample_rate/2
tx_ch.gain = gain_tx

rx_ch = sdr.Channel(_bladerf.CHANNEL_RX(0))
rx_ch.gain_mode = _bladerf.GainMode.Manual
rx_ch.gain = gain_rx
rx_ch.frequency = rx_center_freq
rx_ch.sample_rate = sample_rate
rx_ch.bandwidth = sample_rate/2


bytes_per_sample = 4
rx_buf = bytearray(num_samples*bytes_per_sample)




# Setup synchronous stream
sdr.sync_config(layout=_bladerf.ChannelLayout.TX_X1, # or TX_X2
                fmt=_bladerf.Format.SC16_Q11, # int16s
                num_buffers=8,
                buffer_size=4096,
                num_transfers=4,
                stream_timeout=100000)

sdr.sync_config(layout=_bladerf.ChannelLayout.RX_X1, # or TX_X2
                fmt=_bladerf.Format.SC16_Q11, # int16s
                num_buffers=16,
                buffer_size=8192,
                num_transfers=8,
                stream_timeout=100000)

tx_ch.enable = True
rx_ch.enable = True
# while True:
#     # time.sleep(0.5)
#     try:
#         sdr.sync_tx(buf, num_samples) # write to bladeRF
#     except KeyboardInterrupt:
#         break
print("Transmitting")
sdr.sync_tx(buf, num_samples)
print("Receiving")
sdr.sync_rx(rx_buf, num_samples)

tx_ch.enable = False
sdr.close()

data = np.frombuffer(rx_buf, dtype=np.int16)
data = data[0::2] + 1j * data[1::2] # Convert to complex type
data /= 2048.0

# print(data)

f_carrier = np.linspace(-0.5 * sample_rate, 0.5 * sample_rate, num_samples) + rx_center_freq
data_fft = (20 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(data))) / len(data)))

fig, ax = plt.subplots(2, 1, figsize=(15,9.5))

timedomain, = ax[0].plot(np.linspace(0, 1, num_samples), np.real(data))

ax[0].set_ylim(-1.5, 1.5)

ax[0].set_title("Time Domain RX0")

fftplot, = ax[1].plot(f_carrier, data_fft)

ax[1].set_ylabel("Power (dB)")

ax[1].set_xlabel("Frequency (GHz)")

ax[1].set_title("Frequency Domain RX0")

ax[1].set_ylim(-80, -10)

ax[1].grid()

plt.show()




