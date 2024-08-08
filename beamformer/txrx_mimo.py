import signal
import sys
import threading
import time
from collections.abc import Iterable
from pathlib import Path
from typing import cast

import signal
import sys
import threading
import time
from collections.abc import Iterable
from pathlib import Path
from typing import cast

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from bladerf import _bladerf
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from tqdm import tqdm

from core.utils import generate_bladerf_carrier_signal, fft, MIMObin2complex
from core.sdr import BladeRFWrapper, SimBladeRFWrapper



TX_BLADERF_SERIAL = "104"
RX_BLADERF_SERIAL = "dfc"

CENTER_FREQUENCY = 2.397e9
SAMPLE_RATE = 5e6

TX_GAIN = 66
RX_GAIN = 71

NUM_SAMPLES_PER_CHANNEL = 2048
TX_SAMPLES_BUFFER_SIZE = 2048
RX_SAMPLES_BUFFER_SIZE = TX_SAMPLES_BUFFER_SIZE * 2
BYTES_PER_SAMPLE = 4

ENABLE_TX = sys.argv[1] == "t" if len(sys.argv) == 2 else True
ENABLE_RX = sys.argv[1] == "r" if len(sys.argv) == 2 else True

USE_SIM_SDR = False

def create_sdr(bladerf_serial: str) -> BladeRFWrapper | SimBladeRFWrapper:
    return SimBladeRFWrapper() if USE_SIM_SDR else BladeRFWrapper(bladerf_serial)


def run_tx(tx_sdr: BladeRFWrapper | SimBladeRFWrapper) -> None:
    tx_sdr.enable_tx(
        frequency=CENTER_FREQUENCY-0.5e6,
        sample_rate=SAMPLE_RATE,
        gain=TX_GAIN,
        buffer_size=TX_SAMPLES_BUFFER_SIZE,
    )
    

    while not close_event.is_set():

        complex_samples = generate_bladerf_carrier_signal(num_samples=TX_SAMPLES_BUFFER_SIZE, noise_variance=0.00, scale=2048.0)

        i = complex_samples.real.astype(np.int16)
        q = complex_samples.imag.astype(np.int16)

        iq_samples = np.empty(i.size + q.size, dtype=np.int16)
        iq_samples[0::2] = i
        iq_samples[1::2] = q

        tx_buffer = iq_samples.tobytes()

        for _ in range(200):
            if close_event.is_set():
                break

            tx_sdr.transmit(tx_buffer, len(complex_samples))

    print("Waiting 2 seconds before disabling TX channel.")
    time.sleep(2)
    tx_sdr.disable_tx()


def run_rx(rx_sdr: BladeRFWrapper | SimBladeRFWrapper, plot_once: bool) -> None:

    rx_sdr.enable_rx_mimo(
        frequency=CENTER_FREQUENCY,
        sample_rate=SAMPLE_RATE,
        gain=RX_GAIN,
        num_samples_per_channel=NUM_SAMPLES_PER_CHANNEL,
    )
    rx_buffer_mimo = bytearray(NUM_SAMPLES_PER_CHANNEL * BYTES_PER_SAMPLE * 2)


    fig, ax = plt.subplots(nrows=2, ncols=2,layout="tight")


    timedomainplot_rx0, = ax[0][0].plot(np.zeros(NUM_SAMPLES_PER_CHANNEL))
    frequencydomainplot_rx0, = ax[1][0].plot(np.zeros(NUM_SAMPLES_PER_CHANNEL))

    ax[0][0].set_xlabel("Sample Index")
    ax[0][0].set_ylabel("Amplitude")
    ax[0][0].set_title("RX0 Time Domain RX'ed Data")
    ax[0][0].set_xlim(0, NUM_SAMPLES_PER_CHANNEL)
    ax[0][0].set_ylim(-1, 1)

    ax[1][0].set_xlabel("Frequency")
    ax[1][0].set_ylabel("Power [dB]")
    ax[1][0].set_title("RX0 Frequency Domain RX'ed Data")

    timedomainplot_rx1, = ax[0][1].plot(np.zeros(NUM_SAMPLES_PER_CHANNEL))
    frequencydomainplot_rx1, = ax[1][1].plot(np.zeros(NUM_SAMPLES_PER_CHANNEL))

    ax[0][1].set_xlabel("Sample Index")
    ax[0][1].set_ylabel("Amplitude")
    ax[0][1].set_title("RX1 Time Domain RX'ed Data")
    ax[0][1].set_xlim(0, NUM_SAMPLES_PER_CHANNEL)
    ax[0][1].set_ylim(-1, 1)

    ax[1][1].set_xlabel("Frequency")
    ax[1][1].set_ylabel("Power [dB]")
    ax[1][1].set_title("RX1 Frequency Domain RX'ed Data")

    if(not plot_once):

        def animate(i):

            rx_sdr.receive_mimo(buf=rx_buffer_mimo, num_samples=NUM_SAMPLES_PER_CHANNEL)


            all_complex_samples = MIMObin2complex(rx_buffer_mimo, NUM_SAMPLES_PER_CHANNEL)

            complex_samples = all_complex_samples[0]


            carrier_data = fft(data=complex_samples, 
                            samplerate=SAMPLE_RATE, 
                            center_freq=CENTER_FREQUENCY,
                            freq_of_interest=CENTER_FREQUENCY,
                            plot=False)
            

            timedomainplot_rx0.set_ydata(complex_samples.real)
            frequencydomainplot_rx0.set_xdata(carrier_data[0])
            frequencydomainplot_rx0.set_ydata(carrier_data[1])
            ax[1][0].set_xlim(min(carrier_data[0]), max(carrier_data[0]))
            ax[1][0].set_ylim(-70, 0)

            complex_samples = all_complex_samples[1]

            carrier_data = fft(data=complex_samples, 
                            samplerate=SAMPLE_RATE, 
                            center_freq=CENTER_FREQUENCY,
                            freq_of_interest=CENTER_FREQUENCY,
                            plot=False)


            timedomainplot_rx1.set_ydata(complex_samples.real)
            frequencydomainplot_rx1.set_xdata(carrier_data[0])
            frequencydomainplot_rx1.set_ydata(carrier_data[1])
            ax[1][1].set_xlim(min(carrier_data[0]), max(carrier_data[0]))
            ax[1][1].set_ylim(-70, 0)

            return (timedomainplot_rx0, frequencydomainplot_rx0, timedomainplot_rx1, frequencydomainplot_rx1,)


        anim = animation.FuncAnimation(fig, animate, blit=True, save_count=0,
                                interval=0)
        plt.show()
    else:

        rx_sdr.receive_mimo(rx_buffer_mimo, NUM_SAMPLES_PER_CHANNEL)


        all_complex_samples = MIMObin2complex(rx_buffer_mimo, RX_SAMPLES_BUFFER_SIZE)

        complex_samples = all_complex_samples[0]

        carrier_data = fft(data=complex_samples, 
                        samplerate=SAMPLE_RATE, 
                        center_freq=CENTER_FREQUENCY,
                        freq_of_interest=CENTER_FREQUENCY,
                        plot=False)

        timedomainplot_rx0.set_ydata(complex_samples.real)
        frequencydomainplot_rx0.set_xdata(carrier_data[0])
        frequencydomainplot_rx0.set_ydata(carrier_data[1])
        ax[1][0].set_xlim(min(carrier_data[0]), max(carrier_data[0]))
        ax[1][0].set_ylim(-100, max(carrier_data[1])+10)

        complex_samples = all_complex_samples[1]

        carrier_data = fft(data=complex_samples, 
                        samplerate=SAMPLE_RATE, 
                        center_freq=CENTER_FREQUENCY,
                        freq_of_interest=CENTER_FREQUENCY,
                        plot=False)

        timedomainplot_rx1.set_ydata(complex_samples.real)
        frequencydomainplot_rx1.set_xdata(carrier_data[0])
        frequencydomainplot_rx1.set_ydata(carrier_data[1])
        ax[1][1].set_xlim(min(carrier_data[0]), max(carrier_data[0]))
        ax[1][1].set_ylim(-100, max(carrier_data[1])+10)

        plt.show()    

    close_event.set()
    # rx_sdr.disable_rx()


tx_sdr = create_sdr(TX_BLADERF_SERIAL) if ENABLE_TX else None
rx_sdr = (
    (
        tx_sdr
        if ENABLE_TX and (RX_BLADERF_SERIAL == TX_BLADERF_SERIAL or USE_SIM_SDR)
        else create_sdr(RX_BLADERF_SERIAL)
    )
    if ENABLE_RX
    else None
)

# rx_sdr.print_all_info()

# rx_sdr = create_sdr(RX_BLADERF_SERIAL) if ENABLE_TX else None
# rx_sdr = create_sdr(RX_BLADERF_SERIAL)





if tx_sdr is rx_sdr and isinstance(tx_sdr, BladeRFWrapper):
    print(f"Supported Loopback Modes: {tx_sdr.bladerf.get_loopback_modes()}")
    tx_sdr.bladerf.set_loopback(_bladerf.Loopback.Firmware)
    current_loopback_mode = tx_sdr.bladerf.get_loopback()
    print(f"Set Loopback Mode: {current_loopback_mode}")

    if current_loopback_mode != _bladerf.Loopback.Firmware:
        sys.exit("Failed to set BladeRF loopback to Firmware mode.")

tx_thread = threading.Thread(target=run_tx, args=(tx_sdr,))
close_event = threading.Event()

signal.signal(signal.SIGINT, lambda *_: close_event.set())

if tx_sdr is not None:
    tx_thread.start()

if rx_sdr is not None:
    run_rx(rx_sdr, False)

if tx_sdr is not None:
    tx_thread.join()
    tx_sdr.close()

if rx_sdr is not None and rx_sdr is not tx_sdr:
    rx_sdr.close()

print("Closed SDR.")
