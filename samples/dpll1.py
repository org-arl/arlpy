##########################################################################
# demo program to test PLL
##########################################################################

import numpy as np
import scipy.signal as sp

import arlpy.signal as asig
from arlpy import comms

fs = 1080000.0
fc = 27000.0

x = comms.modulate(comms.random_data(1000, 2), comms.psk(2))
x = comms.upconvert(x, 60, fc, fs, comms.rrcosfir(0.25, 60))

b1, a1 = sp.iirfilter(4, (0.98*2*fc/(fs/2), 1.02*2*fc/(fs/2)))
b2, a2 = sp.iirfilter(4, 0.001, btype='lowpass')
#f1 = asig.lfilter_gen(b1, a1)
f2 = asig.lfilter_gen(b2, a2)
nco = asig.nco_gen(2*fc, fs, phase0=np.pi) #, phase0=2*np.pi*54000*5/fs)

nco.next()
e2 = np.empty_like(x)
#y = np.empty_like(x)
y = sp.lfilter(b1, a1, x**2)
z = np.empty_like(x, dtype=np.complex)
for j in xrange(len(x)):
    #y[j] = f1.send(x[j]**2)
    z[j] = nco.send(2*fc+1e4*(e2[j-1] if j >= 1 else 0))
    e2[j] = f2.send(y[j]*np.real(z[j]))

print np.mean(e2[3000:-3000])
