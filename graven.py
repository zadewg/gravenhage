import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


def banner():

	print("                               __                  ")
	print("  ___ ________ __  _____ ___  / /  ___ ____ ____   ")
	print(" / _ `/ __/ _ `/ |/ / -_) _ \/ _ \/ _ `/ _ `/ -_)  ")
	print(" \_, /_/  \_,_/|___/\__/_//_/_//_/\_,_/\_, /\__/   ")
	print("/___/                                 /___/        ")

	print("\n           Radio Frequency Hacking               ")
	print("\n    https://github.com/zadewg/gravenhage  \n\n   ")



def parsein():

	global LEN, MOD, FREQ, PLOT, SAVE

	parser = argparse.ArgumentParser(description='https://github.com/zadewg/gravenhage')
	parser.add_argument('-kl','--length', help='Register length', required=True)
	parser.add_argument('-u','--use', help='Modulation technique to apply (ASK, FSK, PSK)', required=True)
	parser.add_argument('-f','--freq', help='Carrier signal frequency', required=False)
	parser.add_argument('-p','--plt', help='Signal plotting', action='store_true', required=False)
	parser.add_argument('-of','--outf', help='File to write to', required=False)
	args = vars(parser.parse_args())

	MOD = str(args['use'])
	LEN = int(args['length'])
	FREQ = int(args['freq']) or 10
	PLOT = args['plt'] 
	SAVE = args['outf'] or False



class Modulation:

	"""
	Digital modulation. Suported techniques: Amplitude Shift Keying, Phase Shift Keying, Frequency Shift Keying-
	data: binary string to apply modulation to
	technique: modulation technique to apply {ASK, PSK, FSK}
	frequency: carrier wave frequency
	"""

	def __init__(self, data, technique='ASK', frequency=10):
		self.DATA = data
		self.MOD = technique
		self.FREQ = frequency

		out = self._mod(self.DATA, self.MOD, self.FREQ)
		self.signal = out[0]
		self.spectrum = out[1]
		self.space = out[2]
		self.response = out[3]


	def _mod(self, DATA, MOD, FREQ):

						#FOR PLOTTING PURPOSES	
		Fs = float(len(DATA)*30);  	# sampling rate
		Ts = 1.0/Fs; 			# sampling interval
		t = np.arange(0,2,Ts) 		# sample time

		if (MOD=='FSK'):
			bit_arr = np.array([X*5 if X == 1 else -5 for X in DATA])
			samples_per_bit = 2*Fs/bit_arr.size 
			dd = np.repeat(bit_arr, samples_per_bit)
			y = np.sin(2 * np.pi * (FREQ + dd) * t)

		elif (MOD=='PSK'):
			bit_arr = np.array([X*180 for X in DATA])
			samples_per_bit = 2*Fs/bit_arr.size 
			dd = np.repeat(bit_arr, samples_per_bit)
			y = np.sin(2 * np.pi * (FREQ) * t+(np.pi*dd/180))

		elif (MOD=='ASK'):
			bit_arr = np.array(DATA)
			samples_per_bit = 2*Fs/bit_arr.size 
			dd = np.repeat(bit_arr, samples_per_bit)
			y = dd*np.sin(2 * np.pi * FREQ * t)

		else:
			print("Modulation technique not supported")
			raise SystemExit

		n = len(y) 
		k = np.arange(n)
		T = n/Fs
		frq = k/T 
		frq = frq[range(n//2)]
		Y = np.fft.fft(y)/n 
		Y = Y[range(n//2)]
				
		return y, Y, t, frq



def dbrs(n, k=2):
	
	"""
	k: alphabet space 
	n: subsequence size
	"""

	_mapping = bytearray(b"?")*256
	_mapping[:10] = b"0123456789"

	a = k * n * bytearray([0])
	sequence = bytearray()
	extend = sequence.extend
	def db(t, p):
		if t > n:
			if n % p == 0:
				extend(a[1: p+1])
		else:
			a[t] = a[t - p]
			db(t + 1, p)
			for j in range(a[t - p] + 1, k):
				a[t] = j
				db(t + 1, t)
	db(1, 1)
	return sequence.translate(_mapping).decode("ascii")

banner()
parsein()

payload = dbrs(LEN)
DATA = [int(i) for i in str(payload)]


mod = Modulation(DATA, technique=MOD, frequency=FREQ)

y = mod.signal
Y = mod.spectrum
t = mod.space
frq = mod.response

data = {}
data['sequence'] = DATA
data['encoded'] = y.tolist()

outb = payload if len(payload) <= 57 else payload[0:57]
outm = y if len(y) <= 80 else y[0:80]
print("\nSequence head: {}\n".format(outb))
print("{} Modulated head:\n {}\n".format(MOD, outm))


fig,myplot = plt.subplots(2, 1)
myplot[0].plot(t,y)
myplot[0].set_xlabel('Time')
myplot[0].set_ylabel('Amplitude')

myplot[1].plot(frq,abs(Y),'r')
myplot[1].set_xlabel('Freq (Hz)')
myplot[1].set_ylabel('|Y(freq)|')


if SAVE:
	plt.savefig(SAVE)
	with open(('{}.json'.format(SAVE)), 'w') as outfile:  
		json.dump(data, outfile)

if PLOT:
	plt.show()
