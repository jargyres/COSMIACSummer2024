import numpy as np
import serial
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import threading
import signal
import numpy.typing as npt
from lib.modulation import demodulate_ask, modulate_ask
from lib.packet import create_packet, extract_message
from lib.sdr import BladeRFWrapper, SimBladeRFWrapper
import time
from typing import cast


ser = serial.Serial('/dev/ttyUSB0')
TX_BLADERF_SERIAL = "104b"
RX_BLADERF_SERIAL = "a662"

freq = int(2.5e9)
lookatfreq = int(2505000000)
samplerate = int(20e6)
bandwidth = int(samplerate/2)

rxGain = 55
txGain = 60


angleLimit = 177
# angleLimit = 90


TX_SAMPLES_BUFFER_SIZE = 1024
RX_SAMPLES_BUFFER_SIZE = TX_SAMPLES_BUFFER_SIZE
SAMPLES_PER_SYMBOL = 40
BYTES_PER_SAMPLE = 4



pattern_theta_list = []
pattern_list = []

def generate_signal(
    carrier_freq: float,
    sample_rate: float,
    num_samples: int
) -> npt.NDArray[np.complex128]:

    t = np.linspace(0, num_samples / sample_rate, num_samples)
    carrier_samples: npt.NDArray[np.complex128] = np.exp(
        1j * 2 * np.pi * carrier_freq * t
    )
    return carrier_samples

def run_tx(tx_sdr: BladeRFWrapper | SimBladeRFWrapper) -> None:
    tx_sdr.enable_tx(
        frequency=freq,
        sample_rate=samplerate,
        gain=txGain,
        buffer_size=TX_SAMPLES_BUFFER_SIZE,
    )


    # while True:


    complex_samples = generate_signal(1e3, samplerate, TX_SAMPLES_BUFFER_SIZE)
    scaled_complex_samples = complex_samples * 2047

    i = scaled_complex_samples.real.astype(np.int16)
    q = scaled_complex_samples.imag.astype(np.int16)

    iq_samples = np.empty(i.size + q.size, dtype=np.int16)
    iq_samples[0::2] = i
    iq_samples[1::2] = q

    tx_buffer = iq_samples.tobytes()

    while True:

        if close_event.is_set():
            break

        tx_sdr.transmit(tx_buffer, len(complex_samples))

    print("Waiting 2 seconds before disabling TX channel.")
    time.sleep(2)
    tx_sdr.disable_tx()


def run_rx(rx_sdr: BladeRFWrapper | SimBladeRFWrapper) -> None:
    # rx_sdr.enable_rx(
    #     frequency=lookatfreq,
    #     sample_rate=samplerate,
    #     gain=rxGain,
    #     buffer_size=RX_SAMPLES_BUFFER_SIZE,
    # )

    rx_buffer = bytearray(RX_SAMPLES_BUFFER_SIZE * BYTES_PER_SAMPLE)

    rx_sdr.receive(rx_buffer, RX_SAMPLES_BUFFER_SIZE)

    raw_samples = np.frombuffer(rx_buffer, dtype=np.int16)
    complex_samples = cast(
        npt.NDArray[np.complex128],
        (raw_samples[0::2] + 1j * raw_samples[1::2]) / 2047,
    )
    # rx_sdr.disable_rx()


    return complex_samples


    # close_event.set()



def is_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False
    
def indices(a, func):
    return [i for (i, val) in enumerate(a) if func(val)]

def Get_Combined_RX_Pwr(data):

    # dataC = data[0] + data[1] + data[2] + data[3]



    NUM_SAMPLES = TX_SAMPLES_BUFFER_SIZE

    max_pwr_search_size = 30
    f_carrier = np.linspace(-0.5 * samplerate, 0.5 * samplerate, NUM_SAMPLES) + (lookatfreq)

    data_fft = (20 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(data))) / NUM_SAMPLES))
    carrier_data = [np.transpose(f_carrier), data_fft]
    # carrier_data = np.asanyarray(carrier_data)

    indexes = indices(carrier_data[0], lambda x: x > freq - 3e6 and x < freq + 3e6)

    pwr=max(data_fft[indexes])

    return pwr


def CWReceive(sdr: BladeRFWrapper):
    print("STARTING CLOCKWISE")
    
    data = "CR\n"
    ser.write(data.encode('ascii'))
    data = "CW\n"
    ser.write(data.encode('ascii'))

    pos = ""
    while(True):
        data = "CP?\n"
        ser.write(data.encode('ascii'))
        line = ser.readline()
        l = line.decode('utf-8').rstrip()
        if l:
            if l != "\x06":
                if is_float(l):
                    pos = float(l)
                    rx_samples = run_rx(sdr)
                    pwr_db = Get_Combined_RX_Pwr(rx_samples)

                    pattern_theta_list.append(pos)
                    pattern_list.append(pwr_db)

                    print("{} deg = {} dB".format(pos, pwr_db))


                    if(np.abs(float(l) - angleLimit) < 1.0):
                        break
    while(True):
        data = "ST\n"
        ser.write(data.encode('ascii'))
        data = "DIR?\n"
        ser.write(data.encode('ascii'))
        line = ser.readline()
        l = line.decode('utf-8').rstrip()

        if l:
            if l != "\x06":
                if l == 'N':
                    break

    data = "ST\n"
    ser.write(data.encode('ascii'))

def CCReceive(sdr: BladeRFWrapper):
    print("STARTING COUNTER-CLOCKWISE")
    cc_angle_limit = 360.0 - angleLimit
    data = "CR\n"
    ser.write(data.encode('ascii'))
    data = "CC\n"
    ser.write(data.encode('ascii'))

    pos = ""
    while(True):
        data = "CP?\n"
        ser.write(data.encode('ascii'))
        line = ser.readline()
        l = line.decode('utf-8').rstrip()
        if l:
            if l != "\x06":
                if is_float(l):
                    pos = float(l)
                    rx_samples = run_rx(sdr)
                    pwr_db = Get_Combined_RX_Pwr(rx_samples)

                    pwr_db = Get_Combined_RX_Pwr(rx_samples)

                    pattern_theta_list.append(pos)
                    pattern_list.append(pwr_db)
                    print("{:.4f} deg = {:.4f} dB {:.4f} degrees to go".format(pos - 360.0, pwr_db, (np.abs(float(l) - cc_angle_limit))))

                    if(np.abs(float(l) - cc_angle_limit) < 1.0):
                        break
    while(True):
        data = "ST\n"
        ser.write(data.encode('ascii'))
        data = "DIR?\n"
        ser.write(data.encode('ascii'))
        line = ser.readline()
        l = line.decode('utf-8').rstrip()

        if l:
            if l != "\x06":
                if l == 'N':
                    break

    data = "ST\n"
    ser.write(data.encode('ascii'))

def create_sdr(bladerf_serial: str) -> BladeRFWrapper | SimBladeRFWrapper:
    return BladeRFWrapper(bladerf_serial)


tx_sdr = create_sdr(TX_BLADERF_SERIAL)
rx_sdr = create_sdr(RX_BLADERF_SERIAL)


tx_thread = threading.Thread(target=run_tx, args=(tx_sdr,))

close_event = threading.Event()

signal.signal(signal.SIGINT, lambda *_: close_event.set())

if tx_sdr is not None:
    tx_thread.start()

rx_sdr.enable_rx(
        frequency=lookatfreq,
        sample_rate=samplerate,
        gain=rxGain,
        buffer_size=RX_SAMPLES_BUFFER_SIZE,
    )

CCReceive(rx_sdr)
CWReceive(rx_sdr)

close_event.set()

if tx_sdr is not None:
    tx_thread.join()
    tx_sdr.close()

if rx_sdr is not None and rx_sdr is not tx_sdr:
    rx_sdr.disable_rx()

    rx_sdr.close()

print("Closed SDR.")


# CCReceive()
# CWReceive()

#turn into array
pattern_theta = np.asarray(pattern_theta_list)

for i in range(len(pattern_theta)):
    print(pattern_theta[i])
#turn into radians
pattern_theta = np.deg2rad(pattern_theta)
#turn into array
pattern = np.asarray(pattern_list)
sorted_theta_ind = np.argsort(pattern_theta)

pattern_theta = pattern_theta[sorted_theta_ind]
pattern = pattern[sorted_theta_ind]

pattern, idx = np.unique(pattern, return_index=True)
pattern_theta = pattern_theta[idx]

# max_pwr = np.max(pattern)
# pattern = pattern - max_pwr
fig = plt.figure()
ax = fig.add_subplot(111, polar=True)
ax.set_theta_direction('clockwise')
ax.scatter(pattern_theta, pattern, s=1)
ax.set_thetalim(-np.pi / 2, np.pi/2)
ax.set_thetagrids([-180, -150, -120, -90, -60, -30, 0, 30, 60,90, 120, 150, 180])
ax.set_rticks([0, -4, -8,  -12,  -16,  -20,  -24,  -28,  -32],labels= ["0db", "-4db", "-8db", "-12db", "-16db", "-20db", "-24db", "-28db", "-32db"])
ax.set_theta_zero_location("N")
ax.grid(True,which="minor",linestyle= ":")
ax.grid(True,which="major",linewidth= 1.5)
ax.minorticks_on()
plt.show()