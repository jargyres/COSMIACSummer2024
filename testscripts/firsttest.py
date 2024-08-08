import bladerf
from bladerf import _bladerf
import numpy as np
import matplotlib.pyplot as plt
import struct
import threading

from multiprocessing.pool import ThreadPool




#in Hz
# CENTER_FREQ = int(3e9)
# CENTER_FREQ = int(3e9)

CENTER_FREQ = int(2.350e9)

#in Hz, need this due to DC offset generated at the center frequency
FREQ_OFFSET = int(5e6)
#in Hz
OFFSET_CENTER_FREQ = CENTER_FREQ - FREQ_OFFSET
#in Msps
SAMPLE_RATE = int(40e6)
#in dB
GAIN_RX = 50
#in Hz
BANDWIDTH = int(40e6)

def indices(a, func):
    return [i for (i, val) in enumerate(a) if func(val)]

def Get_RX_Pwr(data):

    # dataC = data[0] + data[1] + data[2] + data[3]
    dataC = data

    # NUM_SAMPLES = 
    NUM_SAMPLES = len(data)

    max_pwr_search_size = 30
    # f = np.linspace(-0.5 * samplerate, 0.5 * samplerate, len(dataC))
    f_carrier = np.linspace(-0.5 * SAMPLE_RATE, 0.5 * SAMPLE_RATE, NUM_SAMPLES) + (OFFSET_CENTER_FREQ)

    data_fft = (20 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(dataC))) / NUM_SAMPLES)) - 20
    carrier_data = [np.transpose(f_carrier), data_fft]
    carrier_data = np.asanyarray(carrier_data)

    indexes = indices(carrier_data[0], lambda x: x > CENTER_FREQ - 1e6 and x < CENTER_FREQ + 1e6)
    # indexes = np.linspace(NUM_SAMPLES/2 - max_pwr_search_size, NUM_SAMPLES/2 + max_pwr_search_size, dtype=int)

    pwr=max(data_fft[indexes])

    # index = np.where(carrier_data == pwr)[0][0]
    index = np.where(data_fft == pwr)[0][0]
    
    return pwr, index

def bin2complex(binary_buffer, complex_array):
    for i in range(0, len(binary_buffer), 4):
        sig_i = struct.unpack('<h', binary_buffer[i:i+2])[0]
        sig_q = struct.unpack('<h', binary_buffer[i+2:i+4])[0]
        complex_array[i//4] = complex(sig_i, sig_q)

def MIMObin2complex(binary_buffer, complex_array):
    index = 0
    for i in range(0, len(binary_buffer), 8):
        sig_i_0 = struct.unpack('<h', binary_buffer[i:i+2])[0]
        sig_q_0 = struct.unpack('<h', binary_buffer[i+2:i+4])[0]

        sig_i_1 = struct.unpack('<h', binary_buffer[i+4:i+6])[0]
        sig_q_1 = struct.unpack('<h', binary_buffer[i+6:i+8])[0]

        complex_array[0][index] = complex(sig_i_0, sig_q_0)
        complex_array[1][index] = complex(sig_i_1, sig_q_1)

        index += 1





def plotRXedData(mimo_buf, mimo_buf_dev2):

    

    mimodata = np.zeros((2, NUM_SAMPLES), dtype=complex)

    MIMObin2complex(mimo_buf, mimodata)

    print(mimodata[0][:5])
    print(mimodata[1][:5])
    # numsamples = NUM_SAMPLES//2

    data = mimodata[0].copy()

    f = np.linspace(-0.5 * SAMPLE_RATE, 0.5 * SAMPLE_RATE, len(data))
    f_carrier = np.linspace(-0.5 * SAMPLE_RATE, 0.5 * SAMPLE_RATE, len(data)) + OFFSET_CENTER_FREQ
    data_fft = (20 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(data))) / len(data))) - 20
    carrier_data = [np.transpose(f_carrier), data_fft]


    fig, ax = plt.subplots(2, 2, figsize=(15,9.5))
    fig2, ax2 = plt.subplots(2, 2, figsize=(15,9.5))

    ax2[0][0].plot(np.linspace(0, 1, len(data)), np.real(data))
    ax2[0][0].set_title("Time Series Master RX_0")
    


    ax[0][0].plot(f_carrier, data_fft)

    ax[0][0].set_ylabel("Power (dB)")

    ax[0][0].set_title("Master RX_0")

    pwr, index = Get_RX_Pwr(data)


    ax[0][0].annotate("{} dB".format(pwr), xy=(f_carrier[index], pwr), xytext=(f_carrier[index], pwr),
                arrowprops=dict(facecolor='black', shrink=0.5),
                bbox=dict(boxstyle="round,pad=0.5", fc="r", alpha=0.5))

    ax[0][0].grid()



    data = mimodata[1].copy()


    f = np.linspace(-0.5 * SAMPLE_RATE, 0.5 * SAMPLE_RATE, len(data))
    f_carrier = np.linspace(-0.5 * SAMPLE_RATE, 0.5 * SAMPLE_RATE, len(data)) + OFFSET_CENTER_FREQ
    data_fft = (20 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(data))) / len(data))) - 20
    carrier_data = [np.transpose(f_carrier), data_fft]

    ax2[0][1].plot(np.linspace(0, 1, len(data)), np.real(data))
    ax2[0][1].set_title("Time Series Master RX_1")

    ax[0][1].plot(f_carrier, data_fft)

    ax[0][1].set_xlabel("Frequency (Hz)")

    ax[0][1].set_ylabel("Power (dB)")

    ax[0][1].set_title("Master RX_1")

    pwr, index = Get_RX_Pwr(data)

    ax[0][1].annotate("{} dB".format(pwr), xy=(f_carrier[index], pwr), xytext=(f_carrier[index], pwr),
                arrowprops=dict(facecolor='black', shrink=0.5),
                bbox=dict(boxstyle="round,pad=0.5", fc="r", alpha=0.5))

    ax[0][1].grid()

    mimodata = np.zeros((2, NUM_SAMPLES), dtype=complex)

    MIMObin2complex(mimo_buf_dev2, mimodata)

    print(mimodata[0][:5])
    print(mimodata[1][:5])

    data = mimodata[0].copy()

    f = np.linspace(-0.5 * SAMPLE_RATE, 0.5 * SAMPLE_RATE, len(data))
    f_carrier = np.linspace(-0.5 * SAMPLE_RATE, 0.5 * SAMPLE_RATE, len(data)) + OFFSET_CENTER_FREQ
    data_fft = (20 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(data))) / len(data))) - 20
    carrier_data = [np.transpose(f_carrier), data_fft]

    ax2[1][0].plot(np.linspace(0, 1, len(data)), np.real(data))
    ax2[1][0].set_title("Time Series Slave RX_0")


    ax[1][0].plot(f_carrier, data_fft)

    ax[1][0].set_ylabel("Power (dB)")

    ax[1][0].set_title("Slave RX_0")

    pwr, index = Get_RX_Pwr(data)

    ax[1][0].annotate("{} dB".format(pwr), xy=(f_carrier[index], pwr), xytext=(f_carrier[index], pwr),
                arrowprops=dict(facecolor='black', shrink=0.5),
                bbox=dict(boxstyle="round,pad=0.5", fc="r", alpha=0.5))

    ax[1][0].grid()

    data = mimodata[1].copy()


    f = np.linspace(-0.5 * SAMPLE_RATE, 0.5 * SAMPLE_RATE, len(data))
    f_carrier = np.linspace(-0.5 * SAMPLE_RATE, 0.5 * SAMPLE_RATE, len(data)) + OFFSET_CENTER_FREQ
    data_fft = (20 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(data))) / len(data))) - 20
    carrier_data = [np.transpose(f_carrier), data_fft]

    ax2[1][1].plot(np.linspace(0, 1, len(data)), np.real(data))
    ax2[1][1].set_title("Time Series Slave RX_1")

    ax[1][1].plot(f_carrier, data_fft)

    ax[1][1].set_xlabel("Frequency (Hz)")

    ax[1][1].set_ylabel("Power (dB)")

    ax[1][1].set_title("Slave RX_1")

    pwr, index = Get_RX_Pwr(data)

    ax[1][1].annotate("{} dB".format(pwr), xy=(f_carrier[index], pwr), xytext=(f_carrier[index], pwr),
                arrowprops=dict(facecolor='black', shrink=0.5),
                bbox=dict(boxstyle="round,pad=0.5", fc="r", alpha=0.5))

    ax[1][1].grid()

    plt.show()


#The real orientation
master_bladerf = bladerf.BladeRF(device_identifier="*:serial=cbd")
slave_bladerf = bladerf.BladeRF(device_identifier="*:serial=35d")

# master_bladerf = bladerf.BladeRF(device_identifier="*:serial=35d")
# slave_bladerf = bladerf.BladeRF(device_identifier="*:serial=cbd")
master_bladerf.set_clock_output(True)
#0 is CLOCK_SELECT_ONBOARD, 1 is CLOCK_SELECT_EXTERNAL
slave_bladerf.set_clock_select(1)





ch_rx_0 = bladerf.CHANNEL_RX(0)
ch_rx_1 = bladerf.CHANNEL_RX(1)
ch_tx_0 = bladerf.CHANNEL_TX(0)
ch_tx_1 = bladerf.CHANNEL_TX(1)



#set both rx to manual gain mode to simplify things
# d.set_gain_mode(ch_rx_0, 1)
# d.set_gain_mode(ch_rx_1, 1)
# master_bladerf.set_gain_mode(ch_rx_0, 0)
# master_bladerf.set_gain_mode(ch_rx_1, 0)
# slave_bladerf.set_gain_mode(ch_rx_0, 0)
# slave_bladerf.set_gain_mode(ch_rx_1, 0)
master_bladerf.set_gain_mode(ch_rx_0, 1)
master_bladerf.set_gain_mode(ch_rx_1, 1)
slave_bladerf.set_gain_mode(ch_rx_0, 1)
slave_bladerf.set_gain_mode(ch_rx_1, 1)

#set the sample rate for the RX chain
master_bladerf.set_sample_rate(ch_rx_0, SAMPLE_RATE)
master_bladerf.set_sample_rate(ch_rx_1, SAMPLE_RATE)
slave_bladerf.set_sample_rate(ch_rx_0, SAMPLE_RATE)
slave_bladerf.set_sample_rate(ch_rx_1, SAMPLE_RATE)

#set bandidth for RX chain
master_bladerf.set_bandwidth(ch_rx_0, BANDWIDTH)
master_bladerf.set_bandwidth(ch_rx_1, BANDWIDTH)
slave_bladerf.set_bandwidth(ch_rx_0, BANDWIDTH)
slave_bladerf.set_bandwidth(ch_rx_1, BANDWIDTH)




#set the gain for the RX chain

master_bladerf.set_gain(ch_rx_0, GAIN_RX)
master_bladerf.set_gain(ch_rx_1, GAIN_RX)
slave_bladerf.set_gain(ch_rx_0, GAIN_RX)
slave_bladerf.set_gain(ch_rx_1, GAIN_RX)

#set center frequency for RX chain
master_bladerf.set_frequency(ch_rx_0, OFFSET_CENTER_FREQ)
master_bladerf.set_frequency(ch_rx_1, OFFSET_CENTER_FREQ)
slave_bladerf.set_frequency(ch_rx_0, OFFSET_CENTER_FREQ)
slave_bladerf.set_frequency(ch_rx_1, OFFSET_CENTER_FREQ)





#Trigger setup
sig = bladerf.TRIGGER_SIGNAL.J51_1.value


masterChannel = bladerf.CHANNEL_TX(0)

masterRole = bladerf.TRIGGER_ROLE.Master.value

masterTrigger = master_bladerf.master_trigger_init(masterChannel, sig)

bladerfslaveChannel_0 = bladerf.CHANNEL_RX(0)

slaveRole = bladerf.TRIGGER_ROLE.Slave.value

master_bladerfslaveTrigger_RX0 = master_bladerf.trigger_init(bladerfslaveChannel_0, slaveRole, sig)

slave_bladerfslaveTrigger_RX0 = slave_bladerf.trigger_init(bladerfslaveChannel_0, slaveRole, sig)


print("Arming All Triggers")
master_bladerf.trigger_arm(masterTrigger, True)
master_bladerf.trigger_arm(master_bladerfslaveTrigger_RX0, True)
slave_bladerf.trigger_arm(slave_bladerfslaveTrigger_RX0, True)




# master_bladerf.sync_config(layout = _bladerf.ChannelLayout.RX_X1,
#                        fmt            = _bladerf.Format.SC16_Q11,
#                        num_buffers    = 16,
#                        buffer_size    = 8192,
#                        num_transfers  = 8,
#                        stream_timeout = 10000)

# master_bladerf.sync_config(layout = _bladerf.ChannelLayout.RX_X2,
#                        fmt            = _bladerf.Format.SC16_Q11,
#                        num_buffers    = 16,
#                        buffer_size    = 2048,
#                        num_transfers  = 8,
#                        stream_timeout = 10000)
# slave_bladerf.sync_config(layout = _bladerf.ChannelLayout.RX_X2,
#                        fmt            = _bladerf.Format.SC16_Q11,
#                        num_buffers    = 16,
#                        buffer_size    = 2048,
#                        num_transfers  = 8,
#                        stream_timeout = 10000)

bytes_per_sample = 4
NUM_SAMPLES = 1024
NUM_CHANNELS = 2
bytesinint16 = 2
buf_size = 2 * NUM_SAMPLES * NUM_CHANNELS * bytesinint16

master_bladerf.sync_config(layout = 2,
                       fmt            = _bladerf.Format.SC16_Q11,
                       num_buffers    = 32,
                       buffer_size    = buf_size,
                       num_transfers  = 8,
                       stream_timeout = 10000)
slave_bladerf.sync_config(layout = 2,
                       fmt            = _bladerf.Format.SC16_Q11,
                       num_buffers    = 32,
                       buffer_size    = buf_size,
                       num_transfers  = 8,
                       stream_timeout = 10000)

print(master_bladerf.get_devinfo())

master_s_0 = master_bladerf.Channel(bladerf.CHANNEL_RX(0))
master_s_0.enable = True
master_s_1 = master_bladerf.Channel(bladerf.CHANNEL_RX(1))
master_s_1.enable = True
slave_s_0 = slave_bladerf.Channel(bladerf.CHANNEL_RX(0))
slave_s_0.enable = True
slave_s_1 = slave_bladerf.Channel(bladerf.CHANNEL_RX(1))
slave_s_1.enable = True




# mimo_buf = bytearray(NUM_SAMPLES*bytes_per_sample * 2)
# mimo_buf_dev2 = bytearray(NUM_SAMPLES*bytes_per_sample * 2)
mimo_buf = bytearray(NUM_SAMPLES*bytes_per_sample * 2)
mimo_buf_dev2 = bytearray(NUM_SAMPLES*bytes_per_sample * 2)

# print(master_bladerf.get_gain(ch_rx_0))
# print(master_bladerf.get_gain(ch_rx_1))
# print(master_bladerf.get_gain_mode(ch_rx_0))
# print(master_bladerf.get_gain_mode(ch_rx_1))


rxpool = ThreadPool(processes=1)
rxpool_dev2 = ThreadPool(processes=1)

result = rxpool.apply_async(master_bladerf.sync_rx, (mimo_buf, NUM_SAMPLES*2))
result_dev2 = rxpool.apply_async(slave_bladerf.sync_rx, (mimo_buf_dev2, NUM_SAMPLES*2))


print("Firing Master Trigger")
master_bladerf.trigger_fire(masterTrigger)

result.wait()
result_dev2.wait()



# Disable module
print( "RX: Stop" )
master_s_0.enable = False
master_s_1.enable = False
slave_s_0.enable = False
slave_s_1.enable = False

print( "RX: Done" )


plotRXedData(mimo_buf, mimo_buf_dev2)



master_bladerf.close()
slave_bladerf.close()

