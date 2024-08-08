import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib.animation import FuncAnimation

sample_rate = 1e6
N = 10000 # number of samples to simulate

# Create a tone to act as the transmitted signal
t = np.arange(N)/sample_rate
f_tone = 0.02e6
tx = np.exp(2j*np.pi*f_tone*t)

# Simulate three omnidirectional antennas in a line with 1/2 wavelength between adjancent ones, receiving a signal that arrives at an angle

d = 0.5
Nr = 4
theta_degrees = 20 # direction of arrival
theta = theta_degrees / 180 * np.pi # convert to radians
#s * tx is what the antenna array will receive
s = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta)) # steering vector
#print(s)

# we have to do a matrix multiplication of s and tx, which currently are both 1D, so we have to make them 2D with reshape
s = s.reshape(-1,1)
#print(s.shape) # 3x1
tx = tx.reshape(1,-1) # make a row vector
# print(tx.shape) # 1x10000

# so how do we use this? simple:
r = s @ tx # matrix multiply
#print(r.shape) # r4x10000.  r is now going to be a 2D array, 1d is time and 1d is spatial

# Plot the real part of the first 200 samples of all three elements
if True:
    fig, (ax1) = plt.subplots(1, 1, figsize=(7, 3))
    ax1.plot(np.asarray(r[0,:]).squeeze().real[0:200]) # the asarray and squeeze are just annoyances we have to do because we came from a matrix
    ax1.plot(np.asarray(r[1,:]).squeeze().real[0:200])
    ax1.plot(np.asarray(r[2,:]).squeeze().real[0:200])
    ax1.plot(np.asarray(r[3,:]).squeeze().real[0:200])
    ax1.set_ylabel("Samples")
    ax1.set_xlabel("Time")
    ax1.grid()
    ax1.legend(['0','1','2', '3'], loc=1)
    plt.show()
    #fig.savefig('../_images/doa_time_domain.svg', bbox_inches='tight')
    # exit()
# note the phase shifts, they are also there on the imaginary portions of the samples

# So far this has been simulating the recieving of a signal from a certain angle of arrival
# in your typical DOA problem you are given samples and have to estimate the angle of arrival(s)
# there are also problems where you have multiple receives signals from different directions and one is the SOI while another might be a jammer or interferer you have to null out

# One thing we didnt both doing- lets add noise to this recieved signal.
# AWGN with a phase shift applied is still AWGN so we can add it after or before the steering vector is applied, doesnt really matter, we'll do it after
# we need to make sure each element gets an independent noise signal added

n = np.random.randn(Nr, N) + 1j*np.random.randn(Nr, N)
r = r + 0.5*n

if True:
    fig, (ax1) = plt.subplots(1, 1, figsize=(7, 3))
    ax1.plot(np.asarray(r[0,:]).squeeze().real[0:200]) # the asarray and squeeze are just annoyances we have to do because we came from a matrix
    ax1.plot(np.asarray(r[1,:]).squeeze().real[0:200])
    ax1.plot(np.asarray(r[2,:]).squeeze().real[0:200])
    ax1.plot(np.asarray(r[3,:]).squeeze().real[0:200])
    ax1.set_ylabel("Samples")
    ax1.set_xlabel("Time")
    ax1.grid()
    ax1.legend(['0','1','2', '3'], loc=1)
    plt.show()
    #fig.savefig('../_images/doa_time_domain_with_noise.svg', bbox_inches='tight')
    # exit()
