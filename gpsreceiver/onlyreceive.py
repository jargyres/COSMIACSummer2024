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
from matplotlib.widgets import Button, Slider




class TX_RX():
    def __init__(self):

        self.utils = Utils()

        self.CENTER_FREQ = int(2.4e9)
        #in Hz, need this due to DC offset generated at the center frequency
        self.FREQ_OFFSET = int(1e6)
        #in Hz
        self.OFFSET_CENTER_FREQ = self.CENTER_FREQ - self.FREQ_OFFSET
        #in Msps
        # self.SAMPLE_RATE = int(20e6)
        self.SAMPLE_RATE = int(4e6)

        # in dB
        self.GAIN_RX = 50

        self.GAIN_TX = 60

        #in Hz
        # self.BANDWIDTH = (self.SAMPLE_RATE/2)
        # self.BANDWIDTH = self.SAMPLE_RATE
        self.BANDWIDTH = int(20e6)

        # self.BANDWIDTH = int(10e6)

        self.num_samples = 8192

        self.sdr = _bladerf.BladeRF(device_identifier="*:serial=a66")
        # self.sdr_tx = _bladerf.BladeRF(device_identifier="*:serial=104")


        #Set up RX Channel information
        self.rx_ch = self.sdr.Channel(_bladerf.CHANNEL_RX(0))
        self.rx_ch.gain_mode = _bladerf.GainMode.Manual
        self.rx_ch.gain = self.GAIN_RX
        self.rx_ch.frequency = self.OFFSET_CENTER_FREQ
        self.rx_ch.sample_rate = self.SAMPLE_RATE
        self.rx_ch.bandwidth = self.BANDWIDTH


        self.bytes_per_sample = 4
        self.buf = bytearray(self.num_samples*self.bytes_per_sample)

        self.sdr.sync_config(layout = _bladerf.ChannelLayout.RX_X1, # or RX_X2
                    fmt = _bladerf.Format.SC16_Q11, # int16s
                    num_buffers    = 16,
                    buffer_size    = 8192,
                    num_transfers  = 8,
                    stream_timeout = 3500)
        

        self.rx_ch.enable = True

    def receive(self):

        self.sdr.sync_rx(self.buf, self.num_samples) # Read into buffer
        self.data = np.frombuffer(self.buf, dtype=np.int16)
        self.data = self.data[0::2] + 1j * self.data[1::2] # Convert to complex type
        self.data /= 2048.0 # Scale to -1 to 1 (its using 12 bit ADC)


    def close_dev(self):
        self.sdr.close()


    
tx_rx = TX_RX()


tx_rx.receive()

f_carrier = np.linspace(-0.5 * tx_rx.SAMPLE_RATE, 0.5 * tx_rx.SAMPLE_RATE, len(tx_rx.data)) + tx_rx.OFFSET_CENTER_FREQ
data_fft = (20 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(tx_rx.data))) / len(tx_rx.data)))

fig, ax = plt.subplots(2, 1, figsize=(15,9.5))

timedomain, = ax[0].plot(np.linspace(0, 1, len(tx_rx.data)), np.imag(tx_rx.data))

ax[0].set_ylim(-1.5, 1.5)

ax[0].set_title("Time Domain RX0")

fftplot, = ax[1].plot(f_carrier, data_fft)

ax[1].set_ylabel("Power (dB)")

ax[1].set_xlabel("Frequency (GHz)")

ax[1].set_title("Frequency Domain RX0")

ax[1].set_ylim(-80, -10)

ax[1].grid()



def slider_update(val):

    SAMPLE_RATE = int(samplerate_slider.val)

    tx_rx.rx_ch.sample_rate = SAMPLE_RATE

    tx_rx.receive()

    f_carrier = np.linspace(-0.5 * tx_rx.SAMPLE_RATE, 0.5 * tx_rx.SAMPLE_RATE, len(tx_rx.data)) + tx_rx.OFFSET_CENTER_FREQ
    
    data_fft = (20 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(tx_rx.data))) / len(tx_rx.data)))
    
    carrier_data = [np.transpose(f_carrier), data_fft]
    
    carrier_data = np.asanyarray(carrier_data)

    ax[0].clear()
    
    ax[0].plot(np.linspace(0, 1, len(tx_rx.data)), np.imag(tx_rx.data))
    
    ax[0].set_ylim(-1.5, 1.5)

    ax[1].clear()

    ax[1].plot(f_carrier, data_fft)

    ax[1].set_ylabel("Power (dB)")

    ax[1].set_xlabel("Frequency (GHz)")

    ax[1].set_title("Frequency Domain RX0")

    ax[1].set_ylim(-80, -10)

    ax[1].grid()



axsamplerate = fig.add_axes([0.25, 0.05, 0.65, 0.03])
samplerate_slider = Slider(
    ax=axsamplerate,
    label='Samplerate [Msps]',
    valmin=520834,
    valmax=61440000,
    valinit=tx_rx.SAMPLE_RATE,
)

samplerate_slider.on_changed(slider_update)


def update(i):


    tx_rx.receive()

    f_carrier = np.linspace(-0.5 * tx_rx.SAMPLE_RATE, 0.5 * tx_rx.SAMPLE_RATE, len(tx_rx.data)) + tx_rx.OFFSET_CENTER_FREQ
    
    data_fft = (20 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(tx_rx.data))) / len(tx_rx.data)))
    
    carrier_data = [np.transpose(f_carrier), data_fft]
    
    carrier_data = np.asanyarray(carrier_data)

    ax[0].clear()
    
    ax[0].plot(np.linspace(0, 1, len(tx_rx.data)), np.imag(tx_rx.data))
    
    ax[0].set_ylim(-1.5, 1.5)

    ax[1].clear()

    ax[1].plot(f_carrier, data_fft)

    ax[1].set_ylabel("Power (dB)")

    ax[1].set_xlabel("Frequency (GHz)")

    ax[1].set_title("Frequency Domain RX0")

    ax[1].set_ylim(-80, -10)

    ax[1].grid()


# ani = animation.FuncAnimation(fig, update, interval=10, cache_frame_data=False)

plt.show()
tx_rx.close_dev()
