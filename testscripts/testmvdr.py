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
Nr = 3
theta_degrees = 20 # direction of arrival
theta = theta_degrees / 180 * np.pi # convert to radians
s = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta)) # steering vector
#print(s)

# we have to do a matrix multiplication of s and tx, which currently are both 1D, so we have to make them 2D with reshape
s = s.reshape(-1,1)
#print(s.shape) # 3x1
tx = tx.reshape(-1,1)
#print(tx.shape) # 10000x1

# so how do we use this? simple:
r = s @ tx.T # matrix multiply. dont get too caught up by the transpose s, the important thing is we're multiplying the steering vector by the tx signal
#print(r.shape) # 3x10000.  r is now going to be a 2D array, 1d is time and 1d is spatial

# One thing we didnt both doing- lets add noise to this recieved signal.
# AWGN with a phase shift applied is still AWGN so we can add it after or before the steering vector is applied, doesnt really matter, we'll do it after
# we need to make sure each element gets an independent noise signal added

n = np.random.randn(Nr, N) + 1j*np.random.randn(Nr, N)
r = r + 0.2*n













# fig, (ax1) = plt.subplots(1, 1, figsize=(7, 3))
# ax1.plot(np.asarray(r[0,:]).squeeze().real[0:200]) # the asarray and squeeze are just annoyances we have to do because we came from a matrix
# ax1.plot(np.asarray(r[1,:]).squeeze().real[0:200])
# ax1.plot(np.asarray(r[2,:]).squeeze().real[0:200])
# ax1.set_ylabel("Samples")
# ax1.set_xlabel("Time")
# ax1.grid()
# ax1.legend(['0','1','2'], loc=1)
# plt.show()

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.set_ylim([-10, 0])
def w_mvdr(theta, r):
    s = np.exp(-2j * np.pi * d * np.arange(r.shape[0]) * np.sin(theta)) # steering vector in the desired direction theta
    s = s.reshape(-1,1) # make into a column vector (size 3x1)
    R = r @ r.conj().T # Calc covariance matrix. gives a Nr x Nr covariance matrix of the samples
    Rinv = np.linalg.pinv(R) # 3x3. pseudo-inverse tends to work better than a true inverse
    w = (Rinv @ s)/(s.conj().T @ Rinv @ s) # MVDR/Capon equation! numerator is 3x3 * 3x1, denominator is 1x3 * 3x3 * 3x1, resulting in a 3x1 weights vector
    return w

def power_mvdr(theta, r):
    s = np.exp(-2j * np.pi * d * np.arange(r.shape[0]) * np.sin(theta)) # steering vector in the desired direction theta_i
    s = s.reshape(-1,1) # make into a column vector (size 3x1)
    R = r @ r.conj().T # Calc covariance matrix. gives a Nr x Nr covariance matrix of the samples
    Rinv = np.linalg.pinv(R) # 3x3. pseudo-inverse tends to work better than a true inverse
    return 1/(s.conj().T @ Rinv @ s).squeeze()

# Nr = 8 # 8 elements
# theta1 = 20 / 180 * np.pi # convert to radians
# # theta2 = 25 / 180 * np.pi
# theta3 = -40 / 180 * np.pi
# s1 = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta1)).reshape(-1,1) # 8x1
# # s2 = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta2)).reshape(-1,1)
# s3 = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta3)).reshape(-1,1)
# # we'll use 3 different frequencies.  1xN
# tone1 = np.exp(2j*np.pi*0.01e6*t).reshape(1,-1)
# tone2 = np.exp(2j*np.pi*0.02e6*t).reshape(1,-1)
# tone3 = np.exp(2j*np.pi*0.03e6*t).reshape(1,-1)
# # r = s1 @ tone1 + s2 @ tone2 + 0.1 * s3 @ tone3
# r = s1 @ tone1 + 0.1 * s3 @ tone3

# n = np.random.randn(Nr, N) + 1j*np.random.randn(Nr, N)
# r = r + 0.05*n # 8xN








def null_steering_vector(r, theta):
    """
    Calculates the null steering vector for MVDR null steering.
    r: Received signal matrix (Nr x N), where Nr is the number of antennas and N is the number of samples.
    theta: Desired beam angle in radians.
    """
    Nr = r.shape[0] # Number of antennas
    d = 1 # Distance between antennas, assuming a uniform linear array
    s = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta)) # Steering vector in the desired direction theta
    R = r @ r.conj().T # Covariance matrix
    Rinv = np.linalg.pinv(R) # Pseudo-inverse of the covariance matrix
    null_vector = Rinv @ s # Null steering vector
    return null_vector

def apply_null_steering(r, theta):
    """
    Applies the null steering vector to the received signal to cancel out the noise.
    r: Received signal matrix (Nr x N), where Nr is the number of antennas and N is the number of samples.
    theta: Desired beam angle in radians.
    """
    print(np.shape(r))
    null_vector = null_steering_vector(r, theta)
    print(np.shape(null_vector.conj()))
    # null_steered_signal = r @ null_vector.conj().T
    null_steered_signal = null_vector.conj().T @ r

    return null_steered_signal



ax.set_ylim([-30, 0])

theta_scan = np.linspace(-1*np.pi, np.pi, 1000) # 1000 different thetas between -180 and +180 degrees
results = []
for theta_i in theta_scan:

    r_weighted = apply_null_steering(r, theta=10)
    power_dB = 10*np.log10(np.var(r_weighted)) # power in signal, in dB so its easier to see small and large lobes at the same time


    # w = w_mvdr(theta_i, r) # 3x1
    # r_weighted = w.conj().T @ r # apply weights
    # power_dB = 10*np.log10(np.var(r_weighted)) # power in signal, in dB so its easier to see small and large lobes at the same time
    results.append(power_dB)
    #results.append(10*np.log10(power_mvdr(theta_i, r))) # compare to using equation for MVDR power, should match, SHOW MATH OF WHY THIS HAPPENS!




results -= np.max(results) # normalize
print(theta_scan[np.argmax(results)] * 180/np.pi) # Angle at peak, in degrees


ax.plot(theta_scan, results) # MAKE SURE TO USE RADIAN FOR POLAR
ax.set_theta_zero_location('N') # make 0 degrees point up
ax.set_theta_direction(-1) # increase clockwise
ax.set_rlabel_position(30)  # Move grid labels away from other labels

ax.set_thetamin(-90)
ax.set_thetamax(90) 

# fig.savefig('../_images/doa_capons.svg', bbox_inches='tight')
#fig.savefig('../_images/doa_capons2.svg', bbox_inches='tight')
plt.show()
# exit()
