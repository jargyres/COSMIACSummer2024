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
# RX_BLADERF_SERIAL = "dfc"

CENTER_FREQUENCY = 2.398e9
SAMPLE_RATE = 20e6

TX_GAIN = 56
RX_GAIN = 55

TX_SAMPLES_BUFFER_SIZE = 1024
RX_SAMPLES_BUFFER_SIZE = TX_SAMPLES_BUFFER_SIZE
BYTES_PER_SAMPLE = 4

ENABLE_TX = sys.argv[1] == "t" if len(sys.argv) == 2 else True
ENABLE_RX = sys.argv[1] == "r" if len(sys.argv) == 2 else True

USE_SIM_SDR = False

def create_sdr(bladerf_serial: str) -> BladeRFWrapper | SimBladeRFWrapper:
    return SimBladeRFWrapper() if USE_SIM_SDR else BladeRFWrapper(bladerf_serial)


def run_tx(tx_sdr: BladeRFWrapper | SimBladeRFWrapper) -> None:
    tx_sdr.enable_tx(
        frequency=CENTER_FREQUENCY,
        sample_rate=SAMPLE_RATE,
        gain=TX_GAIN,
        buffer_size=TX_SAMPLES_BUFFER_SIZE,
    )


    while not close_event.is_set():

        complex_samples = generate_bladerf_carrier_signal(num_samples=TX_SAMPLES_BUFFER_SIZE, noise_variance=0.01, scale=2048.0)

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
    rx_sdr.enable_rx(
        frequency=CENTER_FREQUENCY,
        sample_rate=SAMPLE_RATE,
        gain=RX_GAIN,
        buffer_size=RX_SAMPLES_BUFFER_SIZE,
    )

    rx_buffer = bytearray(RX_SAMPLES_BUFFER_SIZE * BYTES_PER_SAMPLE)


    fig, ax = plt.subplots(nrows=2, ncols=1,layout="tight")


    timedomainplot_rx0, = ax[0].plot(np.zeros(RX_SAMPLES_BUFFER_SIZE))
    frequencydomainplot_rx0, = ax[1].plot(np.zeros(RX_SAMPLES_BUFFER_SIZE))

    ax[0].set_xlabel("Sample Index")
    ax[0].set_ylabel("Amplitude")
    ax[0].set_title("RX0 Time Domain RX'ed Data")
    ax[0].set_xlim(0, RX_SAMPLES_BUFFER_SIZE - 1)
    ax[0].set_ylim(-1, 1)

    ax[1].set_xlabel("Frequency")
    ax[1].set_ylabel("Power [dB]")
    ax[1].set_title("RX0 Frequency Domain RX'ed Data")


    if(not plot_once):


        def animate(i):

            rx_sdr.receive(rx_buffer, RX_SAMPLES_BUFFER_SIZE)

            raw_samples = np.frombuffer(rx_buffer, dtype=np.int16)
            complex_samples = cast(
                npt.NDArray[np.complex128],
                (raw_samples[0::2] + 1j * raw_samples[1::2]) / 2048,
            )

            carrier_data = fft(data=complex_samples, 
                            samplerate=SAMPLE_RATE, 
                            center_freq=CENTER_FREQUENCY,
                            freq_of_interest=CENTER_FREQUENCY,
                            plot=False)

            timedomainplot_rx0.set_ydata(complex_samples.real)
            frequencydomainplot_rx0.set_xdata(carrier_data[0])
            frequencydomainplot_rx0.set_ydata(carrier_data[1])
            ax[1].set_xlim(min(carrier_data[0]), max(carrier_data[0]))
            ax[1].set_ylim(-100, max(carrier_data[1])+10)

            return (timedomainplot_rx0, frequencydomainplot_rx0,)


        anim = animation.FuncAnimation(fig, animate,  blit=True, save_count=0,
                                interval=0)
        plt.show()
    else:

        rx_sdr.receive(rx_buffer, RX_SAMPLES_BUFFER_SIZE)

        raw_samples = np.frombuffer(rx_buffer, dtype=np.int16)
        complex_samples = cast(
            npt.NDArray[np.complex128],
            (raw_samples[0::2] + 1j * raw_samples[1::2]) / 2047,
        )

        carrier_data = fft(data=complex_samples, 
                        samplerate=SAMPLE_RATE, 
                        center_freq=CENTER_FREQUENCY,
                        freq_of_interest=CENTER_FREQUENCY,
                        plot=False)
        

        timedomainplot_rx0.set_ydata(complex_samples.real)
        frequencydomainplot_rx0.set_xdata(carrier_data[0])
        frequencydomainplot_rx0.set_ydata(carrier_data[1])
        ax[1].set_xlim(min(carrier_data[0]), max(carrier_data[0]))
        ax[1].set_ylim(-100, max(carrier_data[1])+10)

        plt.show()    

    close_event.set()
    rx_sdr.disable_rx()


tx_sdr = create_sdr(TX_BLADERF_SERIAL) if ENABLE_TX else None
# rx_sdr = (
#     (
#         tx_sdr
#         if ENABLE_TX and (RX_BLADERF_SERIAL == TX_BLADERF_SERIAL or USE_SIM_SDR)
#         else create_sdr(RX_BLADERF_SERIAL)
#     )
#     if ENABLE_RX
#     else None
# )


# if tx_sdr is rx_sdr and isinstance(tx_sdr, BladeRFWrapper):
#     print(f"Supported Loopback Modes: {tx_sdr.bladerf.get_loopback_modes()}")
#     tx_sdr.bladerf.set_loopback(_bladerf.Loopback.Firmware)
#     current_loopback_mode = tx_sdr.bladerf.get_loopback()
#     print(f"Set Loopback Mode: {current_loopback_mode}")

#     if current_loopback_mode != _bladerf.Loopback.Firmware:
#         sys.exit("Failed to set BladeRF loopback to Firmware mode.")

tx_thread = threading.Thread(target=run_tx, args=(tx_sdr,))
close_event = threading.Event()

signal.signal(signal.SIGINT, lambda *_: close_event.set())

if tx_sdr is not None:
    tx_thread.start()

# if rx_sdr is not None:
#     run_rx(rx_sdr, False)

if tx_sdr is not None:
    tx_thread.join()
    tx_sdr.close()

# if rx_sdr is not None and rx_sdr is not tx_sdr:
#     rx_sdr.close()

print("Closed SDR.")
