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
#print(s.shape) # 4x1
tx = tx.reshape(1,-1) # make a row vector
# print(tx.shape) # 1x10000

# so how do we use this? simple:
r = s @ tx # matrix multiply
#print(r.shape) # r4x10000.  r is now going to be a 2D array, 1d is time and 1d is spatial

# So far this has been simulating the recieving of a signal from a certain angle of arrival
# in your typical DOA problem you are given samples and have to estimate the angle of arrival(s)
# there are also problems where you have multiple receives signals from different directions and one is the SOI while another might be a jammer or interferer you have to null out

# One thing we didnt both doing- lets add noise to this recieved signal.
# AWGN with a phase shift applied is still AWGN so we can add it after or before the steering vector is applied, doesnt really matter, we'll do it after
# we need to make sure each element gets an independent noise signal added

n = np.random.randn(Nr, N) + 1j*np.random.randn(Nr, N)
r = r + 0.5*n

# OK lets use this signal r but pretend we don't know which direction the signal is coming in from, lets try to figure it out
# The "conventional" beamforming approach involves scanning through (sampling) all directions from -pi to +pi (-180 to +180) 
# and at each direction we point the array towards that angle by applying the weights associated with pointing in that direction
# which will give us a single 1D array of samples, as if we recieved it with 1 antenna
# we then calc the mean of the magnitude squared as if we were doing an energy detector
# repeat for a ton of different angles and we can see which angle gave us the max

if False:
    # signal from hack-a-sat 4 where we wanted to find the direction of the least energy because there were jammers
    N = 880 # num samples
    r = np.zeros((Nr,N), dtype=np.complex64)
    r[0, :] = np.fromfile('/home/marc/hackasat4/darkside/dishy/Receiver_0.bin', dtype=np.complex64)
    r[1, :] = np.fromfile('/home/marc/hackasat4/darkside/dishy/Receiver_1.bin', dtype=np.complex64)
    r[2, :] = np.fromfile('/home/marc/hackasat4/darkside/dishy/Receiver_2.bin', dtype=np.complex64)


# conventional beamforming
if True:
    theta_scan = np.linspace(-1*np.pi, np.pi, 1000) # 1000 different thetas between -180 and +180 degrees
    results = []
    for theta_i in theta_scan:
        w = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta_i)) # Conventional, aka delay-and-sum, beamformer
        r_weighted = w.conj().T @ r # apply our weights. remember r is 3x10000
        results.append(10*np.log10(np.var(r_weighted))) # power in signal, in dB so its easier to see small and large lobes at the same time
    results -= np.max(results) # normalize

    # print angle that gave us the max value
    print(theta_scan[np.argmax(results)] * 180 / np.pi) # 19.99999999999998

    fig, (ax1) = plt.subplots(1, 1, figsize=(7, 3))
    ax1.plot(theta_scan*180/np.pi, results) # lets plot angle in degrees
    ax1.plot([20],[np.max(results)],'r.')
    ax1.text(-5, np.max(results) + 0.7, '20 degrees')
    ax1.set_xlabel("Theta [Degrees]")
    ax1.set_ylabel("DOA Metric")
    ax1.grid()
    plt.show()
    #fig.savefig('../_images/doa_conventional_beamformer.svg', bbox_inches='tight')

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(theta_scan, results) # MAKE SURE TO USE RADIAN FOR POLAR
    ax.set_theta_zero_location('N') # make 0 degrees point up
    ax.set_theta_direction(-1) # increase clockwise
    #ax.set_rgrids([0,2,4,6,8]) 
    ax.set_rlabel_position(55)  # Move grid labels away from other labels
    plt.show()
    #fig.savefig('../_images/doa_conventional_beamformer_polar.svg', bbox_inches='tight')

    # exit()

# sweeping angle of arrival
if False:
    theta_txs = np.concatenate((np.repeat(-90, 10), np.arange(-90, 90, 2), np.repeat(90, 10)))
    #theta_txs = [-90]
    theta_scan = np.linspace(-1*np.pi, np.pi, 300)
    results = np.zeros((len(theta_txs), len(theta_scan)))
    for t_i in range(len(theta_txs)):
        print(t_i)

        theta = theta_txs[t_i] / 180 * np.pi
        s = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta))
        s = s.reshape(-1,1) # 3x1
        tone = np.exp(2j*np.pi*0.02e6*t)
        tone = tone.reshape(-1,1) # 10000x1
        r = s @ tone.T

        for theta_i in range(len(theta_scan)):
            w = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta_scan[theta_i]))
            r_weighted = np.conj(w) @ r # apply our weights corresponding to the direction theta_i
            results[t_i, theta_i]  = np.mean(np.abs(r_weighted)**2) # energy detector

    fig, ax = plt.subplots(1, 1, figsize=(8, 5), subplot_kw={'projection': 'polar'})
    fig.subplots_adjust(left=0.025, bottom=0.07, right=0.99, top=0.93, wspace=None, hspace=None) # manually tweaked
    line, = ax.plot(theta_scan, results[0,:])
    ax.set_theta_zero_location('N') # make 0 degrees point up
    ax.set_theta_direction(-1) # increase clockwise
    ax.set_rlabel_position(22.5)  # Move grid labels away from other labels
    text = ax.text(0.6, 12, 'fillmein', fontsize=16)
    ax.text(np.pi/-2, 17, 'endfire →', fontsize=16)
    ax.text(np.pi/2, 12, '← endfire', fontsize=16)
    arrow = ax.arrow(0, 0, 0, 0, head_width=0.1, head_length=1, fc='red', ec='red', lw=2) # doesnt matter what initial coords are
    # Test plot
    if False:
        plt.show()
        exit()
    def update(i):
        i = int(i)
        print(i)
        results_i = results[i,:] / np.max(results[i,:]) * 9 # had to add this in for the last animation because it got too large
        line.set_ydata(results_i)
        d_str = str(np.round(theta_txs[i], 2))
        text.set_text('AoA = ' + d_str + '°')
        arrow.set_xy([[theta_txs[i] / 180 * np.pi, 5], [theta_txs[i] / 180 * np.pi, 0]]) # list of verticies. cant get it to stay an arrow...
        return line, ax
    anim = FuncAnimation(fig, update, frames=np.arange(0, len(theta_txs)), interval=100) # run it through compression https://ezgif.com/optimize after its generated to reduce file size
    anim.save('../_images/doa_sweeping_angle_animation.gif', dpi=80, writer='imagemagick')
    exit()


# varying d animations
if False:
    #ds = np.concatenate((np.repeat(0.5, 10), np.arange(0.5, 4.1, 0.05))) # d is large
    ds = np.concatenate((np.repeat(0.5, 10), np.arange(0.5, 0.02, -0.01))) # d is small
    
    theta_scan = np.linspace(-1*np.pi, np.pi, 1000)
    results = np.zeros((len(ds), len(theta_scan)))
    for d_i in range(len(ds)):
        print(d_i)

        # Have to recalc r
        s = np.exp(-2j * np.pi * ds[d_i] * np.arange(Nr) * np.sin(theta))
        s = s.reshape(-1,1)
        r = s @ tx.T

        # DISABLE FOR THE FIRST TWO ANIMATIONS
        if True:
            theta1 = 20 / 180 * np.pi
            theta2 = -40 / 180 * np.pi
            s1 = np.exp(-2j * np.pi * ds[d_i] * np.arange(Nr) * np.sin(theta1)).reshape(-1,1)
            s2 = np.exp(-2j * np.pi * ds[d_i] * np.arange(Nr) * np.sin(theta2)).reshape(-1,1)
            freq1 = np.exp(2j*np.pi*0.02e6*t).reshape(-1,1)
            freq2 = np.exp(2j*np.pi*-0.02e6*t).reshape(-1,1)
            # two tones at diff frequencies and angles of arrival (not sure it actually had to be 2 diff freqs...)
            r = s1 @ freq1.T + s2 @ freq2.T

        for theta_i in range(len(theta_scan)):
            w = np.exp(-2j * np.pi * ds[d_i] * np.arange(Nr) * np.sin(theta_scan[theta_i]))
            r_weighted = np.conj(w) @ r # apply our weights corresponding to the direction theta_i
            results[d_i, theta_i]  = np.mean(np.abs(r_weighted)**2) # energy detector

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    fig.set_tight_layout(True)
    line, = ax.plot(theta_scan, results[0,:])
    ax.set_thetamin(-90) # only show top half
    ax.set_thetamax(90) 
    ax.set_theta_zero_location('N') # make 0 degrees point up
    ax.set_theta_direction(-1) # increase clockwise
    ax.set_rlabel_position(22.5)  # Move grid labels away from other labels
    text = ax.text(0.6, 12, 'fillmein', fontsize=16)
    def update(i):
        i = int(i)
        print(i)
        results_i = results[i,:] #/ np.max(results[i,:]) * 10 # had to add this in for the last animation because it got too large
        line.set_ydata(results_i)
        d_str = str(np.round(ds[i],2))
        if len(d_str) == 3:
            d_str += '0'
        text.set_text('d = ' + d_str)
        return line, ax
    anim = FuncAnimation(fig, update, frames=np.arange(0, len(ds)), interval=100)
    #anim.save('../_images/doa_d_is_large_animation.gif', dpi=80, writer='imagemagick')
    #anim.save('../_images/doa_d_is_small_animation.gif', dpi=80, writer='imagemagick')
    anim.save('../_images/doa_d_is_small_animation2.gif', dpi=80, writer='imagemagick')
    exit()

# Nr = 3
# d = 0.5
N_fft = 512
theta_degrees = theta
theta_degrees = 20 # there is no SOI, we arent processing samples, this is just the direction we want to point at
thetas = theta_degrees / 180 * np.pi
w = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(thetas)) # conventional beamformer
w = np.conj(w) # or else our answer will be negative/inverted
w_padded = np.concatenate((w, np.zeros(N_fft - Nr))) # zero pad to N_fft elements to get more resolution in the FFT
w_fft_dB = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(w_padded)))**2) # magnitude of fft in dB
w_fft_dB -= np.max(w_fft_dB) # normalize to 0 dB at peak

# Map the FFT bins to angles in radians
theta_bins = np.arcsin(np.linspace(-1, 1, N_fft)) # in radians

# find max so we can add it to plot
theta_max = theta_bins[np.argmax(w_fft_dB)]

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.plot(theta_bins, w_fft_dB) # MAKE SURE TO USE RADIAN FOR POLAR
ax.plot([theta_max], [np.max(w_fft_dB)],'ro')
ax.text(theta_max - 0.1, np.max(w_fft_dB) - 4, np.round(theta_max * 180 / np.pi))
ax.set_theta_zero_location('N') # make 0 degrees point up
ax.set_theta_direction(-1) # increase clockwise
ax.set_rlabel_position(55)  # Move grid labels away from other labels
ax.set_thetamin(-90) # only show top half
ax.set_thetamax(90)
ax.set_ylim([-30, 1]) # because there's no noise, only go down 30 dB
plt.show()
