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

from lib.modulation import demodulate_ask, modulate_ask
from lib.packet import create_packet, extract_payload
from lib.sdr import BladeRFWrapper, SimBladeRFWrapper

# TX_FILE = Path("/home/jta/randomtext.txt")
TX_FILE = Path("/home/jta/bladerf_testing/images/newunm.ppm")

# TX_FILE = Path("/home/jta/known.txt")

BYTES_PER_PACKET = 32


TX_BLADERF_SERIAL = "35d"
RX_BLADERF_SERIAL = "35d"

CENTER_FREQUENCY = 2.4e9
MODULATION_CARRIER_FREQUENCY = 500e3
SAMPLE_RATE = 10e6

TX_GAIN = 60
RX_GAIN = 55

TX_SAMPLES_BUFFER_SIZE = 8192
RX_SAMPLES_BUFFER_SIZE = TX_SAMPLES_BUFFER_SIZE * 2
SAMPLES_PER_SYMBOL = 24
BYTES_PER_SAMPLE = 4

ENABLE_TX = sys.argv[1] == "t" if len(sys.argv) == 2 else True
ENABLE_RX = sys.argv[1] == "r" if len(sys.argv) == 2 else True

USE_SIM_SDR = False

# tx_payload = TX_FILE.read_bytes()

# split_payload = [
#     tx_payload[i : i + BYTES_PER_PACKET]
#     for i in range(0, len(tx_payload), BYTES_PER_PACKET)
# ]

# print(f"Transmitting {len(split_payload)} packets.")

# total_payload_len = len(split_payload)

def create_sdr(bladerf_serial: str) -> BladeRFWrapper | SimBladeRFWrapper:
    return SimBladeRFWrapper() if USE_SIM_SDR else BladeRFWrapper(bladerf_serial)


def run_tx(tx_sdr: BladeRFWrapper | SimBladeRFWrapper) -> None:
    tx_sdr.enable_tx(
        frequency=CENTER_FREQUENCY,
        sample_rate=SAMPLE_RATE,
        gain=TX_GAIN,
        buffer_size=TX_SAMPLES_BUFFER_SIZE,
    )

    tx_payload = TX_FILE.read_bytes()

    ppm_offset_bytes = 15
    tx_payload = tx_payload[ppm_offset_bytes:]
    split_payload = [
        tx_payload[i : i + BYTES_PER_PACKET]
        for i in range(0, len(tx_payload), BYTES_PER_PACKET)
    ]

    print(f"Transmitting {len(split_payload)} packets.")

    while not close_event.is_set():
        for packet_idx, payload_part in enumerate(split_payload):
            data_bits = create_packet(payload_part, packet_idx)
            complex_samples = modulate_ask(
                data_bits, MODULATION_CARRIER_FREQUENCY, SAMPLE_RATE, SAMPLES_PER_SYMBOL
            )
            # print(data_bits)
            # input()
            scaled_complex_samples = complex_samples * 2047

            i = scaled_complex_samples.real.astype(np.int16)
            q = scaled_complex_samples.imag.astype(np.int16)

            iq_samples = np.empty(i.size + q.size, dtype=np.int16)
            iq_samples[0::2] = i
            iq_samples[1::2] = q

            tx_buffer = iq_samples.tobytes()

            if len(complex_samples) > TX_SAMPLES_BUFFER_SIZE:
                raise ValueError("Payload doesn't fit in transmit buffer. Payload Size={}".format(len(complex_samples)))

            for _ in range(200):
                if close_event.is_set():
                    break

                tx_sdr.transmit(tx_buffer, len(complex_samples))

    print("Waiting 2 seconds before disabling TX channel.")
    time.sleep(2)
    tx_sdr.disable_tx()


def run_rx(rx_sdr: BladeRFWrapper | SimBladeRFWrapper) -> None:
    rx_sdr.enable_rx(
        frequency=CENTER_FREQUENCY,
        sample_rate=SAMPLE_RATE,
        gain=RX_GAIN,
        buffer_size=RX_SAMPLES_BUFFER_SIZE,
    )

    rx_buffer = bytearray(RX_SAMPLES_BUFFER_SIZE * BYTES_PER_SAMPLE)

    # fig, ax = cast(tuple[Figure, Axes], plt.subplots(layout="tight"))
    # line = ax.plot(np.zeros(RX_SAMPLES_BUFFER_SIZE))[0]

    # ax.set_xlabel("Sample Index")
    # ax.set_ylabel("Amplitude")
    # ax.set_xlim(0, RX_SAMPLES_BUFFER_SIZE - 1)
    # ax.set_ylim(-1, 1)
    nx = 180
    ny = 271

    fig = plt.figure()
    data = np.zeros((nx, ny, 3))
    im = plt.imshow(data, cmap='gist_gray_r', vmin=0, vmax=1)

    total_received_bytes = b""
    last_packet_idx = -1

    # def plot_rx(frame: int) -> Iterable[Artist]:
    # nonlocal total_received_bytes
    # nonlocal last_packet_idx

    
    def init():
        im.set_data(np.zeros((nx, ny)))

    def animate(i):

        nonlocal total_received_bytes
        nonlocal last_packet_idx

        rx_sdr.receive(rx_buffer, RX_SAMPLES_BUFFER_SIZE)

        raw_samples = np.frombuffer(rx_buffer, dtype=np.int16)
        complex_samples = cast(
            npt.NDArray[np.complex128],
            (raw_samples[0::2] + 1j * raw_samples[1::2]) / 2047,
        )

        # line.set_ydata(complex_samples.real)

        demodulated_bits = demodulate_ask(complex_samples, SAMPLES_PER_SYMBOL)
        packet_idx, received_bytes = extract_payload(demodulated_bits)

        xi = i // ny
        yi = i % ny
        data[xi, yi, 0] = 1
        data[xi, yi, 1] = 1
        data[xi, yi, 2] = 1

        
        # im.set_data(data)

        # last_received_bytes = b""

        if received_bytes and packet_idx != last_packet_idx:
            # if packet_idx < last_packet_idx:
                # print("Received all packets. Writing file.")
                # Path("rx_bytes").write_bytes(total_received_bytes)
                # total_received_bytes = b""

            print(f"Received packet {packet_idx}.")
            # print("Packet ID:{:6} Data:{}".format(packet_idx, received_bytes.decode('UTF-8')))
            # print("Packet ID:{:6} Data:{}".format(packet_idx, received_bytes))
            # bits = []
            int_values = [x for x in received_bytes]
            # print(int_values)
            # for i in range(len(int_values)-2):
                
            
            # print("Hex String: ",received_bytes) 
            # print("Packet {} Integer array: {}".format(packet_idx, int_values)) 
            # print("received index {}: Data:{}".format(packet_idx, received_bytes), end=' ')
            # for c in received_bytes.decode('UTF-8').split():
            #     print(int(c), end=' ')
            #     bits.append(int(c))
            
            # print("")



            total_received_bytes += received_bytes

            if packet_idx > last_packet_idx + 1:
                print("Missed packets.")

            last_packet_idx = packet_idx

        return im

    # while(True):

    #     rx_sdr.receive(rx_buffer, RX_SAMPLES_BUFFER_SIZE)

    #     raw_samples = np.frombuffer(rx_buffer, dtype=np.int16)
    #     complex_samples = cast(
    #         npt.NDArray[np.complex128],
    #         (raw_samples[0::2] + 1j * raw_samples[1::2]) / 2047,
    #     )

    #     # line.set_ydata(complex_samples.real)

    #     demodulated_bits = demodulate_ask(complex_samples, SAMPLES_PER_SYMBOL)
    #     packet_idx, received_bytes = extract_payload(demodulated_bits)

    #     if received_bytes and packet_idx != last_packet_idx:
    #         # if packet_idx < last_packet_idx:
    #             # print("Received all packets. Writing file.")
    #             # Path("rx_bytes").write_bytes(total_received_bytes)
    #             # total_received_bytes = b""

    #         # print(f"Received packet {packet_idx}.")
    #         print("Packet ID:{:6} Data:{}".format(packet_idx, received_bytes.decode('UTF-8')))

    #         total_received_bytes += received_bytes

    #         # if packet_idx > last_packet_idx + 1:
    #             # print("Missed packets.")

    #         last_packet_idx = packet_idx

        # return (line,)

    # _ = animation.FuncAnimation(fig, plot_rx, interval=0, blit=True, save_count=0)
    # plt.show()

    anim = animation.FuncAnimation(fig, animate, init_func=init, save_count=0,
                               interval=0)
    plt.show()

    close_event.set()
    rx_sdr.disable_rx()


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
    run_rx(rx_sdr)

if tx_sdr is not None:
    tx_thread.join()
    tx_sdr.close()

if rx_sdr is not None and rx_sdr is not tx_sdr:
    rx_sdr.close()

print("Closed SDR.")
