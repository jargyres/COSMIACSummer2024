import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import struct

'''
Normalizes complex array between -1 and 1
'''
def normalize_complex_arr(X: npt.NDArray[np.complex128], scale: float):
    a = (np.real(X).min() + np.real(X).max())/2.0
    b = (np.imag(X).min() + np.imag(X).max())/2.0
    Y = X - complex(a, b)
    Y = scale*Y/np.abs(Y).max()
    return Y

def generate_bladerf_carrier_signal(
        num_samples: int,
        noise_variance: float,
        scale: float
    )-> npt.NDArray[np.complex128]:
    i = np.zeros(num_samples)
    q = np.zeros(num_samples)
    for k in range(num_samples):

        #This is how we generate noise for complex-valued signals
        noise_i = np.sqrt(noise_variance/2) * np.random.random()
        noise_q = np.sqrt(noise_variance/2) * np.random.random()

        i[k] = np.cos(2.0 * k * np.pi / num_samples) + noise_i
        q[k] = np.sin(2.0 * k * np.pi / num_samples) + noise_q

    carrier_signal = i + 1j * q

    normalized = normalize_complex_arr(X = carrier_signal, scale=scale)

    return normalized


def indices(a, func):
    return [i for (i, val) in enumerate(a) if func(val)]


def fft(data:npt.NDArray[np.complex128], 
        samplerate: int, 
        center_freq: int, 
        freq_of_interest: int,
        plot: bool):
    

    f_carrier = np.linspace(-0.5 * samplerate, 0.5 * samplerate, len(data)) + center_freq
    #20 * log base 10 will give us the fft scaled to dB values
    data_fft = (20 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(data))) / len(data)))
    carrier_data = [np.transpose(f_carrier), data_fft]
    carrier_data = np.asanyarray(carrier_data)


    if(plot):
        indexes = indices(carrier_data[0], lambda x: x > freq_of_interest - 1e6 and x < freq_of_interest + 1e6)

        pwr=max(data_fft[indexes])

        index = np.where(data_fft == pwr)[0][0]

        fig, ax = plt.subplots(2, 1, figsize=(15,9.5))


        timedomain, = ax[0].plot(np.linspace(0, 1, len(data)), np.real(data))

        ax[0].set_ylim(-1.0, 1.0)

        ax[0].set_title("Time Domain RX0")


        fftplot, = ax[1].plot(f_carrier, data_fft)

        ax[1].set_ylabel("Power (dB)")

        ax[1].set_xlabel("Frequency (GHz)")

        ax[1].set_title("Frequency Domain RX0")

        ax[1].annotate("{} dB".format(pwr), xy=(f_carrier[index], pwr), xytext=(f_carrier[index], pwr),
                    arrowprops=dict(facecolor='black', shrink=0.5),
                    bbox=dict(boxstyle="round,pad=0.5", fc="r", alpha=0.5))

        ax[1].grid()

        plt.show()
    else:
        return carrier_data
    
def MIMObin2complex(binary_buffer, num_samples):
    complex_array = np.zeros((2, num_samples), dtype=np.complex128)

    index = 0
    for i in range(0, len(binary_buffer), 8):
        sig_i_0 = struct.unpack('<h', binary_buffer[i:i+2])[0] / 2048.0
        sig_q_0 = struct.unpack('<h', binary_buffer[i+2:i+4])[0] / 2048.0

        sig_i_1 = struct.unpack('<h', binary_buffer[i+4:i+6])[0] / 2048.0
        sig_q_1 = struct.unpack('<h', binary_buffer[i+6:i+8])[0] / 2048.0

        complex_array[0][index] = complex(sig_i_0, sig_q_0)
        complex_array[1][index] = complex(sig_i_1, sig_q_1)

        index += 1
        

    return complex_array