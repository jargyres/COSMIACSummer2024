from scipy import*
from pylab import*
import matplotlib.pyplot as plt
import array
import numpy as np
import random
from utils import Utils
#intialising binary sequence

'''
n=50
bin_ip_seq = np.random.choice([0, 1], size=(n))
n=len(bin_ip_seq)
print ("The input binary sequence is : ")
print (bin_ip_seq)
print(n)
print("\r")
# utils = Utils()
# bin_ip_seq = utils.gen_ca_codes(10)
# n = len(bin_ip_seq)

#passing it to non return to zero level encoder
NRZ_INP=np.zeros(n)

for i in range(0,n):
    NRZ_INP[i]= 2*bin_ip_seq[i] -1
 
#PN sequence Generator

A = np.array([randint(0,2),randint(0,2),randint(0,2),
randint(0,2),randint(0,2),randint(0,2),randint(0,2)])
print(A)


size = 2**len(A) - 1

PN = []

for i in range(size):
	PN.append(A[-1])
	A = [A[-1]^A[-2], A[0], A[1],A[2], A[3],A[4],A[5]]

for i in range(0,size):
    PN[i]= 2*PN[i] -1
print("NON ZERO TO RETURN PN \r")

# print("NRZ_INP = " = NRZ_INP)

#PN sequence Multiplier
z=[]

for i in range(0,n):
	for j in range(1):
		for k in range(0,size):
				z.append(PN[k]*NRZ_INP[i])
				
		
N=len(z)
print(z)

#BPSK Modulator

Tb=1
t=r_[0:Tb:0.1] 
fc=1
carrier=sqrt(2*(Tb**-1))*sin(2*pi*fc*t) 
M=len(carrier) 
plt.plot(carrier)
plt.title("Carrier")
show()
bpskarray=[]
for i in range(0,N):
  if z[i]>=0:
    bpskarray.append(carrier)
	
  else:
    bpskarray.append((-1*carrier))
bpsksignal=concatenate(bpskarray)
print("bpsksignal")
print(len(bpsksignal))
print(bpsksignal) 

figure(1)
plt.plot(bpsksignal)
xlabel('Time')
ylabel('BPSK Signal')
plt.title("BPSK SIGNAL")

#Jamming signal
# x=[]

# x=rand(n*1270)
# t1=r_[0:n*12.7:0.01]
# Eb=.00001*var(carrier)*len(t1)
# jamming=Eb*x*(sqrt(2*(Tb**-1))*(np.cos(2*np.pi*fc*t1)+1j*(np.sin(2*np.pi*fc*t1))))
# figure(4)
# plt.plot(jamming)
# plt.title("Jamming")
# print("\n Jamming")
# print (jamming)
# jamvar=10**(-0.1*(jamming))

# print("varience of carrier\n")
# print(Eb)
# snr=Eb*((jamvar)**-1)
# print(snr)



# snrdb=10*log10(snr)
# print("\nsnrdb")
# print(snrdb)
# print(len(snrdb))



#Reception

receivedsignal=np.array(bpsksignal)
receivearr=[]
print("\n Received signal")
print(receivedsignal)

ber=[]

for i in range(50):
	# noise2=sqrt(i)*randn(len(bpsksignal))
	recsig=bpsksignal
	receivearr=[]
	
	for j in range(N):
		out = sum(carrier*recsig[j*M:(j+1)*M])
		
		#print(out)		
		if out>0:
			receivearr.append(1)
		else:
			receivearr.append(-1)
	rbit=receivearr

	berate=sum(abs(np.array(z)-rbit))*(n**-1)
	print(berate)
	ber.append(berate)
print("\nDemodulated signal")
print(receivearr)
print("\r")
w=len(receivearr)

#PN Squence Multiplication
q=[] 
x=0
for i in range(int(N/size)):
	for k in range(size):	
			q.append(PN[k]*receivearr[x])
			x+=1

print("\nPN multiplied")
print (q)
print(len(q))

print("\r")
#Final array generation
ber=[]
finalarr=[]
for i in range(0,len(q),size):
    if q[i]==-1:
        finalarr.append(0)
    else:
        finalarr.append(1)


	
print("\nBER")
print(ber)
print(len(ber))

print("\nFinal arary")	
print(finalarr);	
print(len(finalarr))
print("\nbinary input")
print(bin_ip_seq)


figure(1)
plt.plot(bpsksignal)
xlabel('Time')
ylabel('BPSK Signal')
plt.title("BPSK SIGNAL")


# figure(2)
# plt.plot(t1,ber, 'bo',t1,ber,'k')
# xlabel('SNR')
# ylabel('BER')
# plt.plot(snr[25000:25050],ber)
# show()

#check whether input sequence in same as received seq
for i in range(n):
	if (finalarr[i]==bin_ip_seq[i]):
		c=1
	else:
		c=0
		break

if(c==1):
	print("\nyes")
else:
	print("NO")

# print(len(snr))
# print(len(jamming))
# print(len(ber))
# print(len(carrier))
# print(len(q))


plt.show()
'''



def generate_sin_wave(frequency: int):
    # Set the carrier frequency and message frequency
    # Generate the time array
    t = np.arange(0, 1, 1/(2_400_000))

    # Generate the carrier signal
    c = np.sin(2 * np.pi * frequency * t)
    return c


mhz = 1_000_000
carrier_wave_frequency = int(157542 * 10e3)
carrier_wave_frequency = int(157542 * 10e3)
print(carrier_wave_frequency)
wave = generate_sin_wave(carrier_wave_frequency)
#plt.figure(figsize=(200,6))
#yyplt.xlim((carrier_wave_frequency - mhz, carrier_wave_frequency + mhz))
plt.xlim((0, mhz/1000))
plt.plot(wave)
plt.title("Carrier Wave")
plt.show()

# Set the carrier frequency and message frequency
fc = int(157542 * 10e3)

print(fc)
fm = 5

# Generate the time array
t = np.arange(0, 1, 1/1000000)

# Generate the carrier signal
c = np.sin(2 * np.pi * fc * t)

# Generate the message signal
m = np.sin(2 * np.pi * fm * t)

# Generate the AM signal
am = c * (2 + m)

# Plot the AM signal
plt.plot(am, 'r')
plt.title("AM Signal")
plt.show()
#plt.figure(figsize=(2000, 6), dpi=80)
plt.figure()

carrier_wave_frequency = int(157542 * 10e3)
print(carrier_wave_frequency)
wave = generate_sin_wave(carrier_wave_frequency)
plt.plot(wave)
plt.title("1575 MHz Carrier")
plt.show()
# Chipping frequency
prn_chip_frequency = 10.23e6 / 10

if False:
	# Plot PRN chips
	#chip = np.array([0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1])
	chip = np.array([0, 0, 0, 0, 1, 1, 1, 1])
	#adjusted_to_freq = [chip[int(x * prn_chip_frequency) % 16] for x in range(20)]
	for x in range(20):
		print(f'x * chip freq: {x * prn_chip_frequency}')
		print(f'idx: {int(x * prn_chip_frequency) % len(chip)}')
		print(f'val: {chip[int(x * prn_chip_frequency) % len(chip)]}')
	adjusted_to_freq = [chip[int(x * prn_chip_frequency) % len(chip)] for x in range(20)]
	print(adjusted_to_freq)
	plt.step(np.arange(0, len(adjusted_to_freq)), adjusted_to_freq)

word =  [1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1]

# for proper scaling
#const = 48000

graph_data_x = []
graph_data_y = []
for i in range(len(word)):
	graph_data_x.append(i * prn_chip_frequency)
	graph_data_x.append((i + 1) * prn_chip_frequency)
	graph_data_y.append(word[i])
	graph_data_y.append(word[i])


plt.plot(graph_data_x, graph_data_y)
plt.title("PRN Code")

plt.show()
