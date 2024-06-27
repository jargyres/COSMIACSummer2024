import bladerf
from bladerf import _bladerf
import numpy as np
import matplotlib.pyplot as plt
import struct
import threading
import matplotlib.animation as animation
from timeit import default_timer as timer
import scipy.signal as signal
from scipy.signal import butter, lfilter
from multiprocessing.pool import ThreadPool
from utils import Utils



class TX_RX():
    def __init__(self):

        self.utils = Utils()

        self.CENTER_FREQ = int(2.4e9)
        #in Hz, need this due to DC offset generated at the center frequency
        self.FREQ_OFFSET = int(5e6)
        #in Hz
        self.OFFSET_CENTER_FREQ = self.CENTER_FREQ - self.FREQ_OFFSET
        #in Msps
        self.SAMPLE_RATE = int(20e6)
        #in dB
        self.GAIN_RX = 60

        self.GAIN_TX = 60

        #in Hz
        self.BANDWIDTH = self.SAMPLE_RATE/2

        self.num_samples = 2048

        self.sdr = _bladerf.BladeRF(device_identifier="*:serial=a66")
        # self.sdr_tx = _bladerf.BladeRF(device_identifier="*:serial=104")


        #Set up RX Channel information
        self.rx_ch = self.sdr.Channel(_bladerf.CHANNEL_RX(0))
        self.rx_ch.gain_mode = _bladerf.GainMode.Manual
        self.rx_ch.gain = self.GAIN_RX
        self.rx_ch.frequency = self.OFFSET_CENTER_FREQ
        self.rx_ch.sample_rate = self.SAMPLE_RATE
        self.rx_ch.bandwidth = self.BANDWIDTH


        #Set up TX Channel information
        self.tx_ch = self.sdr_tx.Channel(_bladerf.CHANNEL_TX(0))
        self.tx_ch.frequency = self.CENTER_FREQ
        self.tx_ch.sample_rate = self.SAMPLE_RATE
        self.tx_ch.bandwidth = self.BANDWIDTH
        self.tx_ch.gain = self.GAIN_TX

        self.bytes_per_sample = 4
        self.buf = bytearray(self.num_samples*self.bytes_per_sample)

        '''
        # self.data = np.frombuffer(self.buf, dtype=np.int16)
        # self.data = self.data[0::2] + 1j * self.data[1::2] # Convert to complex type
        # self.data /= 2048.0

        # self.tx_t = np.arange(self.num_samples) / self.SAMPLE_RATE
        # self.f_tone = 1e6
        # self.tx_samples_v1 = np.exp(1j * 2 * np.pi * self.f_tone * self.tx_t) # will be -1 to +1
        # self.tx_samples = self.tx_samples_v1.astype(np.complex64)
        # self.tx_samples *= 32767 # scale so they can be stored as int16s
        # self.tx_samples = self.tx_samples.view(np.int16)
        # self.tx_buf = self.tx_samples.tobytes()


        # N = 1024

        # fs = self.SAMPLE_RATE

        # fc = self.CENTER_FREQ

        # ts = 1 / float(fs)

        # t = np.arange(0, N * ts, ts)

        # i = np.cos(2 * np.pi * t * fc) * 2 ** 14

        # q = np.sin(2 * np.pi * t * fc) * 2 ** 14

        # self.tx_buf = i + 1j * q

        # input_arr = np.random.randint(2, size=8)
        # input_arr = np.array([1,1, 0, 0, 1, 1, 0,0])
        '''

        start_binary = np.array([1,1,1,1,1,1,1,0])
        real_data = np.random.randint(2, size=8)
        input_arr = np.append(start_binary,real_data)

        print(input_arr)

        # input_arr += 1
        input_arr *= 5

        samples_per_symbol = 128

        phase_signal = np.repeat(input_arr, samples_per_symbol)




        i = np.zeros(self.num_samples)
        q = np.zeros(self.num_samples)
        for k in range(self.num_samples):
            i[k] = np.round(2048.0 * np.cos(2.0 * k * np.pi / self.num_samples))
            q[k] = np.round(2048.0 * np.sin(2.0 * k * np.pi / self.num_samples))
            if(i[k] > 2047):
                i[k] = 2047
            if(i[k] < -2048):
                i[k] = -2048
            if(q[k] > 2047):
                q[k] = 2047
            if(i[k] < -2048):
                q[k] = -2048          

        carrier_signal = i + 1j * q

        # self.tx_buf = phase_signal * carrier_signal
        self.tx_buf = phase_signal


        plt.plot(self.tx_buf)
        plt.show()

        '''
        # self.tx_buf = self.utils.gen_bpsk_signal(input_arr, self.SAMPLE_RATE)

        # self.tx_buf = self.tx_buf.astype(np.complex64)
        # self.tx_buf = self.tx_buf.view(np.int16)

        # print(self.tx_buf)
        # self.tx_samples = self.tx_samples_v1.astype(np.complex64)
        # self.tx_samples *= 32767 # scale so they can be stored as int16s
        # self.tx_samples = self.tx_samples.view(np.int16)
        # self.tx_buf = self.tx_buf.tobytes()

        # return self.iq
        # self.tx_t = np.arange(self.num_samples) / self.SAMPLE_RATE
        # self.f_tone = 1e6
        # self.tx_samples_v1 = np.exp(1j * 2 * np.pi * self.f_tone * self.tx_t) # will be -1 to +1
        # self.tx_samples = self.tx_samples_v1.astype(np.complex64)
        # self.tx_samples *= 32767 # scale so they can be stored as int16s
        # self.tx_samples = self.tx_samples.view(np.int16)
        # self.tx_buf = self.tx_samples.tobytes()

        '''

        self.sdr_tx.sync_config(layout=_bladerf.ChannelLayout.TX_X1, # or TX_X2
                        fmt=_bladerf.Format.SC16_Q11, # int16s
                        num_buffers=16,
                        buffer_size=8192,
                        num_transfers=8,
                        stream_timeout=10000)
        
        self.sdr.sync_config(layout = _bladerf.ChannelLayout.RX_X1, # or RX_X2
                    fmt = _bladerf.Format.SC16_Q11, # int16s
                    num_buffers    = 16,
                    buffer_size    = 8192,
                    num_transfers  = 8,
                    stream_timeout = 3500)
        
        self.tx_ch.enable = True

        self.rx_ch.enable = True

    def transmit(self):
        self.sdr_tx.sync_tx(self.tx_buf, self.num_samples) # write to bladeRF

    def receive(self):

        self.sdr.sync_rx(self.buf, self.num_samples) # Read into buffer
        self.data = np.frombuffer(self.buf, dtype=np.int16)
        self.data = self.data[0::2] + 1j * self.data[1::2] # Convert to complex type
        self.data /= 2048.0 # Scale to -1 to 1 (its using 12 bit ADC)


    def close_dev(self):
        self.sdr.close()


    
tx_rx = TX_RX()

# from lms import LMS

# lms = LMS(step=2, filter_order=1023)
# lowcut = 2e6
# highcut = 19.9999e6


# tx_rx.set_butter_filter(lowcut, highcut, tx_rx.SAMPLE_RATE)

tx_rx.transmit()
tx_rx.receive()

# tx_rx.filter()



# lowcut = tx_rx.nysquist_freq(tx_rx.CENTER_FREQ-100e6)
# highcut = tx_rx.nysquist_freq(tx_rx.CENTER_FREQ+100e6)
# print("lowcut = {} highcut = {} lowcut<highcut = {}".format(lowcut, highcut, lowcut < highcut))


# data = tx_rx.bandpass(tx_rx.data, [tx_rx.nysquist_freq(20e6), tx_rx.nysquist_freq(10e6)], float(tx_rx.SAMPLE_RATE))
# filtered = tx_rx.butter_bandpass(tx_rx.data, lowcut, highcut, tx_rx.SAMPLE_RATE)




f_carrier = np.linspace(-0.5 * tx_rx.SAMPLE_RATE, 0.5 * tx_rx.SAMPLE_RATE, len(tx_rx.data)) + tx_rx.OFFSET_CENTER_FREQ
data_fft = (20 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(tx_rx.data))) / len(tx_rx.data)))

fig, ax = plt.subplots(2, 1, figsize=(15,9.5))

timedomain, = ax[0].plot(np.linspace(0, 1, len(tx_rx.data)), np.imag(tx_rx.data))

ax[0].set_ylim(-1.0, 1.0)

ax[0].set_title("Time Domain RX0")

fftplot, = ax[1].plot(f_carrier, data_fft)

ax[1].set_ylabel("Power (dB)")

ax[1].set_xlabel("Frequency (GHz)")

ax[1].set_title("Frequency Domain RX0")

ax[1].set_ylim(-80, -10)

ax[1].grid()



def update(i):

    tx_rx.transmit()

    tx_rx.receive()

    # tx_rx.utils.bpsk_demodulate(tx_rx.data)

    # tx_rx.data = tx_rx.filter(tx_rx.data)

    f_carrier = np.linspace(-0.5 * tx_rx.SAMPLE_RATE, 0.5 * tx_rx.SAMPLE_RATE, len(tx_rx.data)) + tx_rx.OFFSET_CENTER_FREQ
    
    data_fft = (20 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(tx_rx.data))) / len(tx_rx.data)))
    
    carrier_data = [np.transpose(f_carrier), data_fft]
    
    carrier_data = np.asanyarray(carrier_data)

    ax[0].clear()
    
    ax[0].plot(np.linspace(0, 1, len(tx_rx.data)), np.imag(tx_rx.data))
    
    ax[0].set_ylim(-1.0, 1.0)

    ax[1].clear()

    ax[1].plot(f_carrier, data_fft)

    ax[1].set_ylabel("Power (dB)")

    ax[1].set_xlabel("Frequency (GHz)")

    ax[1].set_title("Frequency Domain RX0")

    ax[1].set_ylim(-80, -10)

    ax[1].grid()


ani = animation.FuncAnimation(fig, update, interval=100, cache_frame_data=False)

plt.show()

tx_rx.close_dev()