import numpy as np
import scipy.special
import matplotlib.pyplot as plt

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from arlpy.common import *
import arlpy.signal as asig
from arlpy import comms

## BPSK

def tx_bpsk(d):
    return comms.modulate(d, comms.psk())

def rx_bpsk(x):
    return comms.demodulate(x, comms.psk())

bpsk = { 'name': 'BPSK', 'tx': tx_bpsk, 'rx': rx_bpsk, 'm': 2, 'n': 1 }

## QPSK

def tx_qpsk(d):
    return comms.modulate(d, comms.psk(4))

def rx_qpsk(x):
    return comms.demodulate(x, comms.psk(4))

qpsk = { 'name': 'QPSK', 'tx': tx_qpsk, 'rx': rx_qpsk, 'm': 4, 'n': 1 }

## BFSK

def tx_bfsk(d):
    return comms.modulate(d, comms.fsk())

def rx_bfsk(x):
    return comms.demodulate(x, comms.fsk())

bfsk = { 'name': 'BFSK', 'tx': tx_bfsk, 'rx': rx_bfsk, 'm': 2, 'n': 2 }

## MSK

def tx_msk(d):
    return comms.modulate(d, comms.msk())

def rx_msk(x):
    return comms.demodulate(x, comms.msk())

msk = { 'name': 'MSK', 'tx': tx_msk, 'rx': rx_msk, 'm': 4, 'n': 4 }

## tests

tests = [bpsk, qpsk, bfsk, msk]

## run tests

fs = 108000
fd = 18000
fc = 27000
fstop = 0.6*fd
ebn0 = range(0, 21, 1)
bits = 100000
pb = False

def run_test(t, ebn0=range(21), syms=1000000):
    ber = []
    for s in ebn0:
        d1 = comms.random_data(syms, t['m'])
        x = t['tx'](d1)
        assert np.abs(np.std(x)-1) < 0.2
        if pb:
            x = asig.bb2pb(x, fd, fc, fs)
            x = comms.awgn(x, s-pow2db(0.5*fs/(2*fstop))-pow2db(t['n'])+pow2db(np.log2(t['m'])))
            x = asig.pb2bb(x, fs, fc, fd)
        else:
            x = comms.awgn(x, s-pow2db(t['n'])+pow2db(np.log2(t['m'])))
        d2 = t['rx'](x)
        ber.append(comms.ber(d1, d2, t['m']))
    return ebn0, ber

def Q(x):
    return 0.5*scipy.special.erfc(x/np.sqrt(2))

plt.figure()

names = []
for t in tests:
    ebn0, ber = run_test(t, ebn0, int(bits/np.log2(t['m'])))
    plt.semilogy(ebn0, ber, '*')
    names.append(t['name'])

ebn0 = np.arange(0, 21, 0.1)
plt.plot(ebn0, Q(db2mag(ebn0+3)), '--')
names.append('BPSK/QPSK theory')
plt.plot(ebn0, 2*Q(db2mag(ebn0)), '--')  # FIXME: fudge factor of 2, why???
names.append('MSK theory')

plt.axis([0, 20, 1e-5, 0.5])
plt.xlabel('Eb/N0 (dB)')
plt.ylabel('BER')
plt.grid()
plt.legend(names)
plt.show()
