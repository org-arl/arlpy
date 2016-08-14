##########################################################################
# demo program to simulate BER for PSK and FSK and compare against theory
##########################################################################

import numpy as np
import scipy.special
import matplotlib.pyplot as plt

from arlpy import comms
from arlpy.utils import db2mag, pow2db

## settings

tests = [
    { 'name': 'BPSK', 'const': comms.psk(2) },
    { 'name': 'QPSK', 'const': comms.psk(4) },
    { 'name': 'BFSK', 'const': comms.fsk(2) }
]

ebn0 = range(0, 21, 1)
bits = 100000

pb = False
fs = 108000
fd = 18000
fc = 27000

## run tests

if pb:
    assert fs % fd == 0
    fs_by_fd = fs/fd
    pulse = comms.rrcosfir(0.25, fs_by_fd)
    pulse_delay = (len(pulse)-1)/2/fs_by_fd

def run_test(const, ebn0=range(21), syms=1000000):
    m = len(const)
    n = const.shape[1] if const.ndim == 2 else 1
    ber = []
    for s in ebn0:
        # eb is divided across n samples, and each symbol has m bits
        snr_per_sample = s + pow2db(np.log2(m)/n)
        d1 = comms.random_data(syms, m)
        x = comms.modulate(d1, const)
        if pb:
            x = comms.upconvert(x, fs_by_fd, fc, fs, pulse)
            # 3 dB extra SNR per sample needed to account for conjugate noise spectrum
            x = comms.awgn(x, snr_per_sample+3, complex=False)
            x = comms.downconvert(x, fs_by_fd, fc, fs, pulse)
            x = x[2*pulse_delay:-2*pulse_delay]
        else:
            x = comms.awgn(x, snr_per_sample, complex=True)
        d2 = comms.demodulate(x, const)
        ber.append(comms.ber(d1, d2, m))
    return ebn0, ber

plt.figure()

names = []
for t in tests:
    ebn0, ber = run_test(t['const'], ebn0, int(bits/np.log2(len(t['const']))))
    plt.semilogy(ebn0, ber, '*')
    names.append(t['name'])

## theory

def Q(x):
    return 0.5*scipy.special.erfc(x/np.sqrt(2))

ebn0 = np.arange(0, 21, 0.1)
plt.plot(ebn0, Q(np.sqrt(2)*db2mag(ebn0)), '--')
names.append('BPSK/QPSK theory')
plt.plot(ebn0, Q(db2mag(ebn0)), '--')
names.append('BFSK theory')

plt.axis([0, 20, 1e-5, 0.5])
plt.xlabel('Eb/N0 (dB)')
plt.ylabel('BER')
plt.grid()
plt.legend(names)
plt.show()
