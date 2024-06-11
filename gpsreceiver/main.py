import numpy as np
import matplotlib.pyplot as plt
# # from utils import Utils

# # def psk_modulation(bits, samples_per_symbol):
# #     modulation_order = 2
# #     # Define the phase states for the chosen modulation order
# #     phase_states = np.linspace(0, 2*np.pi, modulation_order, endpoint=False)

# #     # Map the bits to the corresponding phase states
# #     phase_sequence = [phase_states[int(b)] for b in bits]

# #     # Generate the continuous phase signal with the desired samples per symbol
# #     phase_signal = np.repeat(phase_sequence, samples_per_symbol)

# #     # Generate the carrier signal
# #     t = np.arange(0,len(phase_signal))/10e3
# #     fc = 0.5e3
# #     carrier_signal = np.sin(2*np.pi*fc*t+phase_signal)

# #     return carrier_signal


# # # Example inputs
# # bits = [0, 1, 0, 1, 1, 0]  # Sequence of bits
# # # bits = utils.gen_ca_codes(20)
# # samples_per_symbol = 20  # Number of samples per symbol
# # # samples_per_symbol = 1

# # # Compute the PSK signal
# # psk_signal = psk_modulation(bits, samples_per_symbol)




# # # plt.plot(psk_signal)

# # # plt.show()

# # SAMPLE_RATE = int(40e6)
# # OFFSET_CENTER_FREQ = int(1.57e9)

# # f_carrier = np.linspace(-0.5 * SAMPLE_RATE, 0.5 * SAMPLE_RATE, 1024) + OFFSET_CENTER_FREQ

# # import random
# # import numpy as np
# # from numpy import sin, pi

# # from matplotlib import rcParams
# # # import matplotlib.pylab as plt

# # # f = 25000
# # # Define parameters
# # fs = int(40e6)  # Sampling frequency in Hz (e.g., 10 kHz)
# # N = 1024    # Length of the signal in samples
# # f = int(1.023e6)     # Frequency of the carrier signal in Hz (e.g., 440 Hz)


# # # Generate time array
# # T = np.linspace(0, N/fs, N, False)  # 1/frequency

# # # Create the signal
# # signal = np.sin(2 * np.pi * f * T)
# # prn_seq = utils.gen_ca_codes(20)
# # signal = signal[:-1] * prn_seq
# # # Plot the signal
# # plt.figure(figsize=(10, 6))
# # plt.plot(T[:-1], signal)
# # plt.title('Signal at {} Hz'.format(f))
# # plt.xlabel('Time [s]')
# # plt.ylabel('Amplitude')
# # plt.grid(True)
# # plt.show()





# # for i in range(16):
# #     if(prn_seq[i] == 0):
# #         prn_seq[i] = -1

# # # print(prn_seq)

# # f = 25e3
# # f_prn =  25e3 /10

# # carrier = lambda t: sin(2*pi*f*t)
# # def prn_np(t):
# #     return [ prn_seq[int(ti*f_prn)%16] for ti in t]

# # #t=np.linspace(0,8/f_prn,400)
# # t=np.linspace(0,16/f_prn,1400)
# # plt.figure(figsize=(16,6))
# # # plt.plot(t,prn_np(t),color='red',lw=4)
# # plt.plot(t,signal(t),'b',lw=4,alpha=0.6)
# # plt.title('CDMA BPSK',size=20)
# # plt.xlim([0,t[-1]])
# # plt.ylim([-1.5,1.5])
# # plt.xlabel('Time[s]',size=12)
# # plt.grid()
# # plt.show()


# import matplotlib.pyplot as plt
# import numpy as num
# A=5;
# t=num.arange(0,1,0.001)
# #print(t)
# # f1=int(input('Carrier Sine wave frequency ='))
# # f2=int(input('Message frequency ='))
# f1 = int(10e6)
# f2 = 10
# x=A*num.sin(2*num.pi*f1*t)

# plt.plot(t,x);
# plt.xlabel("time");
# plt.ylabel("Amplitude");
# plt.title("Carrier");
# plt.grid(True)
# plt.show()

# u=[]#Message signal
# b=[0.2,0.4,0.6,0.8,1.0]
# s=1
# for i in t:
#     if(i==b[0]):
#         b.pop(0)
#         if(s==0):
#             s=1
#         else:
#             s=0
#         #print(s,i,b)
#     u.append(s)

# #print(u)

# plt.plot(t,u)
# plt.xlabel('time')
# plt.ylabel('Amplitude')
# plt.title('Message Signal')
# plt.grid(True)
# plt.show()

# v=[]#Sine wave multiplied with square wave
# for i in range(len(t)):
#     if(u[i]==1):
#         v.append(A*num.sin(2*num.pi*f1*t[i]))
#     else:
#         v.append(A*num.sin(2*num.pi*f1*t[i])*-1)

# plt.plot(t,v);
# #plt.axis([0 1 -6 6]);
# plt.xlabel("t");
# plt.ylabel("y");
# plt.title("PSK");
# plt.grid(True)
# plt.show()
# '''Enter the frequency of carrier=10
# Enter the frequency of pulse=2'''

# Sampling frequency  
fs = int(40e6)  
# Carrier frequency  
fc = int(1.57e9)  
# Modulation frequency  
# fm = 100  
# # Modulation index  
# beta = 5  
# Time  
t = np.arange(0, 1, 1/fs)
carrier = np.sin(2*np.pi*fc*t)

plt.plot(carrier[:1024])
plt.show()