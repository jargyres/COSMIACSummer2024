import bladerf
from bladerf import _bladerf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button, Slider


def indices(a, func):
    return [i for (i, val) in enumerate(a) if func(val)]


def plotRXedData(data):


    # f = np.linspace(-0.5 * SAMPLE_RATE, 0.5 * SAMPLE_RATE, len(data))
    f_carrier = np.linspace(-0.5 * SAMPLE_RATE, 0.5 * SAMPLE_RATE, len(data)) + OFFSET_CENTER_FREQ
    #20 * log base 10 will give us the fft scaled to dB values
    data_fft = (20 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(data))) / len(data))) - 20
    carrier_data = [np.transpose(f_carrier), data_fft]
    carrier_data = np.asanyarray(carrier_data)


    indexes = indices(carrier_data[0], lambda x: x > CENTER_FREQ - 1e6 and x < CENTER_FREQ + 1e6)

    pwr=max(data_fft[indexes])

    # index = np.where(carrier_data == pwr)[0][0]
    index = np.where(data_fft == pwr)[0][0]

    fig, ax = plt.subplots(2, 1, figsize=(15,9.5))



    ax[0].plot(np.linspace(0, 1, len(data)), np.real(data))
    ax[0].set_title("Time Domain RX0")
    


    ax[1].plot(f_carrier, data_fft)

    ax[1].set_ylabel("Power (dB)")

    ax[1].set_xlabel("Frequency (Hz)")


    ax[1].set_title("Frequency Domain RX0")

    ax[1].annotate("{} dB".format(pwr), xy=(f_carrier[index], pwr), xytext=(f_carrier[index], pwr),
                arrowprops=dict(facecolor='black', shrink=0.5),
                bbox=dict(boxstyle="round,pad=0.5", fc="r", alpha=0.5))

    ax[1].grid()

    plt.show(block=False)

    fig = plt.figure()
    plt.scatter(np.real(data), np.imag(data))
    plt.title("IQ Data RX0")
    plt.ylabel("Imag")
    plt.xlabel("Real")

    plt.show()



#in Hz
# CENTER_FREQ = int(1.57e9)
CENTER_FREQ = int(2.402e9)

# CENTER_FREQ = int(2.350e9)

#in Hz, need this due to DC offset generated at the center frequency
FREQ_OFFSET = int(5e6)
#in Hz
OFFSET_CENTER_FREQ = CENTER_FREQ - FREQ_OFFSET
#in Msps
SAMPLE_RATE = int(40e6)
#in dB
GAIN_RX = 60
#in Hz
BANDWIDTH = int(40e6)

NUM_SAMPLES = 1024

dev = bladerf.BladeRF(device_identifier="*:serial=cbd")

ch_rx = dev.Channel(bladerf.CHANNEL_RX(0))


ch_rx.frequency = OFFSET_CENTER_FREQ
ch_rx.sample_rate = SAMPLE_RATE
ch_rx.bandwidth = BANDWIDTH
ch_rx.gain_mode = _bladerf.GainMode.Manual
ch_rx.gain = GAIN_RX


dev.sync_config(layout = 0, # or RX_X2
                fmt = _bladerf.Format.SC16_Q11, # int16s
                num_buffers    = 16,
                buffer_size    = 8192,
                num_transfers  = 8,
                stream_timeout = 3500)

ch_rx.enable = True

# Create receive buffer
bytes_per_sample = 4
# num_samples = 1024
buf = bytearray(NUM_SAMPLES*bytes_per_sample)

dev.sync_rx(buf, NUM_SAMPLES)

fig, ax = plt.subplots(2, 1, figsize=(15,9.5))
# plt.tight_layout()


data = np.frombuffer(buf, dtype=np.int16)
data = data[0::2] + 1j * data[1::2] # Convert to complex type
data /= 2048.0

# f = np.linspace(-0.5 * SAMPLE_RATE, 0.5 * SAMPLE_RATE, len(data))
f_carrier = np.linspace(-0.5 * SAMPLE_RATE, 0.5 * SAMPLE_RATE, len(data)) + OFFSET_CENTER_FREQ
#20 * log base 10 will give us the fft scaled to dB values
data_fft = (20 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(data))) / len(data))) - 20
carrier_data = [np.transpose(f_carrier), data_fft]
carrier_data = np.asanyarray(carrier_data)


indexes = indices(carrier_data[0], lambda x: x > CENTER_FREQ - 1e6 and x < CENTER_FREQ + 1e6)

pwr=max(data_fft[indexes])

# index = np.where(carrier_data == pwr)[0][0]
index = np.where(data_fft == pwr)[0][0]

# fig, ax = plt.subplots(2, 1, figsize=(15,9.5))


# ax[0].clear()
timedomain, = ax[0].plot(np.linspace(0, 1, len(data)), np.real(data))

# ax[0].set_ylim(-1.0, 1.0)

ax[0].set_title("Time Domain RX0")


# ax[1].clear()
fftplot, = ax[1].plot(f_carrier, data_fft)
# ax[1].set_data(f_carrier, data_fft)

# ax[1].set_ylim(-100, -40)
# ax[1].set_xticks(np.arange(OFFSET_CENTER_FREQ - BANDWIDTH/2, OFFSET_CENTER_FREQ + BANDWIDTH/2, 9))
# ax[1].set_ylim(f_carrier[0], f_carrier[-1])

# ax[1].set_xlim(OFFSET_CENTER_FREQ - BANDWIDTH/2, OFFSET_CENTER_FREQ + BANDWIDTH/2)

ax[1].set_ylabel("Power (dB)")

ax[1].set_xlabel("Frequency (GHz)")

ax[1].set_title("Frequency Domain RX0")

ax[1].annotate("{} dB".format(pwr), xy=(f_carrier[index], pwr), xytext=(f_carrier[index], pwr),
            arrowprops=dict(facecolor='black', shrink=0.5),
            bbox=dict(boxstyle="round,pad=0.5", fc="r", alpha=0.5))

ax[1].grid()
# ax[1].set_yscale('log')


# plt.show(block=False)

# # fig = plt.figure()
# plt.scatter(np.real(data), np.imag(data))
# plt.title("IQ Data RX0")
# plt.ylabel("Imag")
# plt.xlabel("Real")

# plt.show()


plt.subplots_adjust(bottom=0.3)

# axfreq = fig.add_axes([0.25, 0, 0.65, 0.03])
# freq_slider = Slider(
#     ax=axfreq,
#     label='Frequency [Hz]',
#     valmin=70000000,
#     valmax=6000000000,
#     valinit=2400000000,
# )

axsamplerate = fig.add_axes([0.25, 0.05, 0.65, 0.03])
samplerate_slider = Slider(
    ax=axsamplerate,
    label='Samplerate [Msps]',
    valmin=520834,
    valmax=61440000,
    valinit=40000000,
)

axbandwidth = fig.add_axes([0.25, 0.1, 0.65, 0.03])
bandwidth_slider = Slider(
    ax=axbandwidth,
    label='Bandwidth [Hz]',
    valmin=200000,
    valmax=56000000,
    valinit=40000000,
    # valinit=200000,

)

# xticks = np.arange(OFFSET_CENTER_FREQ - (BANDWIDTH/2), OFFSET_CENTER_FREQ + (BANDWIDTH/2))
# ax[1].set_xticks(xticks)
# # dev.sync_rx(buf, NUM_SAMPLES)

# # # Disable module
# # # ch_rx.enable = False

# # samples = np.frombuffer(buf, dtype=np.int16)
# # samples = samples[0::2] + 1j * samples[1::2] # Convert to complex type
# # samples /= 2048.0

# # plotRXedData(samples)

def slider_update(val):
    # CENTER_FREQ = int(freq_slider.val)


    # OFFSET_CENTER_FREQ = CENTER_FREQ - FREQ_OFFSET

    SAMPLE_RATE = int(samplerate_slider.val)

    BANDWIDTH = int(bandwidth_slider.val)

    # ch_rx.frequency = OFFSET_CENTER_FREQ
    # ch_rx.sample_rate = SAMPLE_RATE
    # ch_rx.bandwidth = BANDWIDTH
    # ch_rx.gain_mode = _bladerf.GainMode.Manual
    # ch_rx.gain = GAIN_RX

    ch_rx_enable = False

    ch_rx.frequency = OFFSET_CENTER_FREQ
    ch_rx.sample_rate = SAMPLE_RATE
    ch_rx.bandwidth = BANDWIDTH
    ch_rx.gain_mode = _bladerf.GainMode.Manual
    ch_rx.gain = GAIN_RX

    dev.sync_config(layout = 0, # or RX_X2
                fmt = _bladerf.Format.SC16_Q11, # int16s
                num_buffers    = 16,
                buffer_size    = 8192,
                num_transfers  = 8,
                stream_timeout = 3500)

    ch_rx.enable = True


    # print(f_carrier[0])


    # ax[1].set_xticks(np.arange(OFFSET_CENTER_FREQ - BANDWIDTH/2, OFFSET_CENTER_FREQ + BANDWIDTH/2, 9))



# freq_slider.on_changed(slider_update)
samplerate_slider.on_changed(slider_update)
bandwidth_slider.on_changed(slider_update)


def update(i):

    dev.sync_rx(buf, NUM_SAMPLES)

    data = np.frombuffer(buf, dtype=np.int16)
    data = data[0::2] + 1j * data[1::2] # Convert to complex type
    data /= 2048.0

    # f = np.linspace(-0.5 * SAMPLE_RATE, 0.5 * SAMPLE_RATE, len(data))
    f_carrier = np.linspace(-0.5 * SAMPLE_RATE, 0.5 * SAMPLE_RATE, len(data)) + OFFSET_CENTER_FREQ
    #20 * log base 10 will give us the fft scaled to dB values
    data_fft = (20 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(data))) / len(data))) - 20
    carrier_data = [np.transpose(f_carrier), data_fft]
    carrier_data = np.asanyarray(carrier_data)


    indexes = indices(carrier_data[0], lambda x: x > CENTER_FREQ - 1e6 and x < CENTER_FREQ + 1e6)

    pwr=max(data_fft[indexes])

    index = np.where(data_fft == pwr)[0][0]


    timedomain.set_data(np.linspace(0, 1, len(data)), np.real(data))


    # ax[0].set_xlim(left=0, right=3)
    # plt.draw()
    # ax[0].clear()
    # ax[0].plot(np.linspace(0, 1, len(data)), np.real(data))
    
    # ax[0].set_ylim(-1.0, 1.0)

    # ax[0].set_title("Time Domain RX0")
    # ax[1].set_xlim(left=OFFSET_CENTER_FREQ, right=OFFSET_CENTER_FREQ)
    
    fftplot.set_data(f_carrier, data_fft)


    # plt.draw()

    # # ax[1].set_data(f_carrier, data_fft)

    # ax[1].set_ylim(-100, -40)
    # ax[1].set_xticks(np.arange(OFFSET_CENTER_FREQ - BANDWIDTH/2, OFFSET_CENTER_FREQ + BANDWIDTH/2, 9))
    # # ax[1].set_ylim(f_carrier[0], f_carrier[-1])

    # ax[1].set_xlim(OFFSET_CENTER_FREQ - BANDWIDTH/2, OFFSET_CENTER_FREQ + BANDWIDTH/2)

    # ax[1].set_ylabel("Power (dB)")

    # ax[1].set_xlabel("Frequency (GHz)")

    # ax[1].set_title("Frequency Domain RX0")

    # ax[1].annotate("{} dB".format(pwr), xy=(f_carrier[index], pwr), xytext=(f_carrier[index], pwr),
    #             arrowprops=dict(facecolor='black', shrink=0.5),
    #             bbox=dict(boxstyle="round,pad=0.5", fc="r", alpha=0.5))

    # ax[1].grid()
    # # ax[1].set_yscale('log')


    # # plt.show(block=False)

    # # # fig = plt.figure()
    # # plt.scatter(np.real(data), np.imag(data))
    # # plt.title("IQ Data RX0")
    # # plt.ylabel("Imag")
    # # plt.xlabel("Real")

    # plt.show()
    return timedomain, fftplot


ani = animation.FuncAnimation(fig, update, interval=100, cache_frame_data=False)


plt.show()

dev.close()