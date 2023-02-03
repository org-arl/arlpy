##############################################################################
#
# Copyright (c) 2016, Mandar Chitre
#
# This file is part of arlpy which is released under Simplified BSD License.
# See file LICENSE or go to http://www.opensource.org/licenses/BSD-3-Clause
# for full license details.
#
##############################################################################

import unittest
import numpy as np
import scipy.signal as sp

from .context import utils, geo, uwa, signal, comms, bf

class MyTestCase(unittest.TestCase):

    def assertApproxEqual(self, x, y, precision=0):
        self.assertEqual(np.round(x, decimals=precision), np.round(y, decimals=precision))

    def assertArrayEqual(self, a, b, msg='', precision=None):
        if precision is None:
            np.testing.assert_array_equal(a, b, err_msg=msg)
        else:
            np.testing.assert_almost_equal(a, b, decimal=precision, err_msg=msg)

class UtilsTestSuite(MyTestCase):

    def test_dB_conversions(self):
        self.assertEqual(utils.mag2db(10.0), 20.0)
        self.assertEqual(utils.db2mag(20.0), 10.0)
        self.assertEqual(utils.pow2db(100.0), 20.0)
        self.assertEqual(utils.db2pow(20.0), 100.0)

    def test_linspace(self):
        x = utils.linspace2d(0, 1, 2, 0, 1, 3)
        self.assertArrayEqual(x, [[0, 0], [0, 0.5], [0, 1], [1, 0], [1, 0.5], [1, 1]])
        x = utils.linspace3d(0, 1, 2, 0, 1, 3, 0, 0, 1)
        self.assertArrayEqual(x, [[0, 0, 0], [0, 0.5, 0], [0, 1, 0], [1, 0, 0], [1, 0.5, 0], [1, 1, 0]])

    def test_rotation_matrix(self):
        x = utils.rotation_matrix(0, 0, 0)
        self.assertArrayEqual(x, [[1, 0, 0], [0, 1, 0], [0, 0, 1]], precision=6)
        x = utils.rotation_matrix(np.pi/2, 0, 0)
        self.assertArrayEqual(x, [[1, 0, 0], [0, 0, -1], [0, 1, 0]], precision=6)
        x = utils.rotation_matrix(np.pi, 0, 0)
        self.assertArrayEqual(x, [[1, 0, 0], [0, -1, 0], [0, 0, -1]], precision=6)
        x = utils.rotation_matrix(0, np.pi/2, 0)
        self.assertArrayEqual(x, [[0, 0, 1], [0, 1, 0], [-1, 0, 0]], precision=6)
        x = utils.rotation_matrix(0, np.pi, 0)
        self.assertArrayEqual(x, [[-1, 0, 0], [0, 1, 0], [0, 0, -1]], precision=6)
        x = utils.rotation_matrix(0, 0, np.pi/2)
        self.assertArrayEqual(x, [[0, -1, 0], [1, 0, 0], [0, 0, 1]], precision=6)
        x = utils.rotation_matrix(0, 0, np.pi)
        self.assertArrayEqual(x, [[-1, 0, 0], [0, -1, 0], [0, 0, 1]], precision=6)

    def test_broadcastable_to(self):
        x = np.array([1,2,3])
        y = utils.broadcastable_to(x, (2,3,4), 1)
        self.assertArrayEqual(y, [[1],[2],[3]])
        y = utils.broadcastable_to(x, (3,2,4), 0)
        self.assertArrayEqual(y, [[[1]],[[2]],[[3]]])
        y = utils.broadcastable_to(x, (2,4,3), 2)
        self.assertArrayEqual(y, [1,2,3])
        y = utils.broadcastable_to(x, (2,4,3), -1)
        self.assertArrayEqual(y, [1,2,3])
        y = utils.broadcastable_to(x, (2,4,3))
        self.assertArrayEqual(y, [1,2,3])
        y = utils.broadcastable_to(x, (2,3,4))
        self.assertArrayEqual(y, [[1],[2],[3]])

    def test_progress(self):
        # no regression test, since this is a display utility function
        pass

class GeoTestSuite(MyTestCase):

    def test_pos(self):
        self.assertEqual(list(map(round, geo.pos([1, 103, 20]))), [277438, 110598, 20])
        self.assertEqual(geo.zone([1, 103]), (48, 'N'))
        self.assertEqual(geo.zone([1, 103, 20]), (48, 'N'))
        x = (1.25, 103.5, 10.0)
        y = geo.latlong(geo.pos(x), geo.zone(x))
        self.assertEqual(tuple(np.round(y, 5)), x)
        self.assertEqual(geo.d(x), (1.25, 103.5))
        self.assertEqual(geo.dm(x), (1.0, 15.0, 103.0, 30.0))
        self.assertEqual(geo.dms(x), (1.0, 15.0, 0.0, 103.0, 30.0, 0))
        self.assertEqual(geo.dz(x), (1.25, 103.5, 10.0))
        self.assertEqual(geo.dmz(x), (1.0, 15.0, 103.0, 30.0, 10.0))
        self.assertEqual(geo.dmsz(x), (1.0, 15.0, 0.0, 103.0, 30.0, 0.0, 10.0))
        p1 = [100.0, 200.0, -5.0]
        p2 = [400.0, 600.0, -5.0]
        self.assertEqual(geo.distance(p1, p1), 0.0)
        self.assertEqual(geo.distance(p1, p2), 500.0)
        self.assertEqual(list(map(round, geo.pos([1, 103, 20], origin=(1, 103)))), [0, 0, 20])
        org = (1, 103)
        y = geo.latlong(geo.pos(x, origin=org), origin=org)
        self.assertEqual(tuple(np.round(y, 5)), x)

class UwaTestSuite(MyTestCase):

    def test_soundspeed(self):
        self.assertApproxEqual(uwa.soundspeed(27, 35, 10), 1539)

    def test_absorption(self):
        self.assertApproxEqual(utils.mag2db(uwa.absorption(10000, temperature=15)), -1)
        self.assertApproxEqual(utils.mag2db(uwa.absorption(50000)), -11)
        self.assertApproxEqual(utils.mag2db(uwa.absorption(100000)), -36)
        self.assertApproxEqual(utils.mag2db(uwa.absorption(100000, temperature=14, salinity=38.5, depth=0)), -40)
        self.assertApproxEqual(utils.mag2db(uwa.absorption(100000, temperature=14, salinity=38.5, depth=2000)), -30)
        self.assertApproxEqual(utils.mag2db(uwa.absorption(100000, temperature=14, salinity=38.5, depth=6000)), -16)

    def test_absorption_filter(self):
        b = uwa.absorption_filter(200000)
        w, h = sp.freqz(b, 1, 4)
        h = utils.mag2db(np.abs(h))
        self.assertEqual(list(np.round(h)), [0.0, -3.0, -11.0, -22.0])

    def test_density(self):
        self.assertApproxEqual(uwa.density(27, 35), 1023)

    def test_reflection(self):
        self.assertApproxEqual(uwa.reflection_coeff(0, 1200, 1600, 0, 1023, 1540), 0.0986, precision=4)
        self.assertApproxEqual(uwa.reflection_coeff(0.5, 1200.0, 1600.0, 0.2, 1023, 1540), 0.0855-0.1278j, precision=4)

    def test_doppler(self):
        self.assertEqual(uwa.doppler(0, 50000), 50000)
        self.assertApproxEqual(uwa.doppler(10, 50000), 50325)
        self.assertApproxEqual(uwa.doppler(-10, 50000), 49675)

    def test_bubble_resonance(self):
        self.assertApproxEqual(uwa.bubble_resonance(100e-6), 32465.562964469198)
        self.assertApproxEqual(uwa.bubble_resonance(32e-6), 101454.88426396625)
        self.assertApproxEqual(uwa.bubble_resonance(100e-6, depth=10), 45796.45437634176)

    def test_bubble_surface_loss(self):
        self.assertApproxEqual(utils.mag2db(uwa.bubble_surface_loss(15, 20000, 1.396)), -6.5, precision=1)
        self.assertApproxEqual(utils.mag2db(uwa.bubble_surface_loss(10, 20000, 1.396)), -3.4, precision=1)
        self.assertApproxEqual(utils.mag2db(uwa.bubble_surface_loss(5, 20000, 1.396)), -0.5, precision=1)

    def test_bubble_soundspeed(self):
        self.assertApproxEqual(uwa.bubble_soundspeed(0, 1500), 1500)
        self.assertApproxEqual(uwa.bubble_soundspeed(1e-5, 1500), 1372)
        self.assertApproxEqual(uwa.bubble_soundspeed(1, 1500, 330), 330)

    def test_pressure(self):
        nbits = 16
        V_ref = 1.0
        x_volt = V_ref*signal.cw(64, 1, 512)
        x_bit = x_volt*(2**(nbits-1))
        sensitivity = 0
        gain = 0
        p1 = uwa.pressure(x_volt, sensitivity, gain)
        p2 = uwa.pressure(x_bit, sensitivity, gain, volt_params=(nbits, V_ref))
        self.assertArrayEqual(p1, p2)

    def test_spl(self):
        p = signal.cw(64, 1, 512)
        self.assertApproxEqual(uwa.spl(p), 20*np.log10(1/np.sqrt(2)))

class SignalTestSuite(MyTestCase):

    def test_time(self):
        self.assertArrayEqual(signal.time(1000, 500), np.arange(1000)/500.0)
        self.assertArrayEqual(signal.time(np.zeros(1000), 500), np.arange(1000)/500.0)

    def test_cw(self):
        self.assertArrayEqual(signal.cw(10000, 0.1, 50000), np.sin(2*np.pi*10000*np.arange(5000, dtype=np.float)/50000), precision=6)
        self.assertArrayEqual(signal.cw(10000, 0.1, 50000, complex_output=True), np.exp(2j*np.pi*10000*np.arange(5000, dtype=np.complex)/50000), precision=6)
        self.assertArrayEqual(signal.cw(10000, 0.1, 50000, ('tukey', 0.1)), sp.tukey(5000, 0.1)*np.sin(2*np.pi*10000*np.arange(5000, dtype=np.float)/50000), precision=2)

    def test_sweep(self):
        self.assertArrayEqual(signal.sweep(5000, 10000, 0.1, 50000), sp.chirp(np.arange(5000, dtype=np.float)/50000, 5000, 0.1, 10000, 'linear'))
        self.assertArrayEqual(signal.sweep(5000, 10000, 0.1, 50000, 'hyperbolic'), sp.chirp(np.arange(5000, dtype=np.float)/50000, 5000, 0.1, 10000, 'hyperbolic'))
        self.assertArrayEqual(signal.sweep(5000, 10000, 0.1, 50000, window=('tukey', 0.1)), sp.tukey(5000, 0.1)*sp.chirp(np.arange(5000, dtype=np.float)/50000, 5000, 0.1, 10000), precision=2)

    def test_envelope(self):
        x = np.random.normal(0, 1, 1000)
        self.assertArrayEqual(signal.envelope(x), np.abs(sp.hilbert(x)))

    def test_mseq(self):
        # we only test until 16, as longer sequences are too slow!
        for j in range(2, 17):
            x = signal.mseq(j)
            self.assertArrayEqual(np.abs(x), np.ones(len(x)))
            x_fft = np.fft.fft(x)
            y = np.fft.ifft(x_fft*x_fft.conj()).real
            self.assertApproxEqual(y[0], len(x), precision=6)
            self.assertArrayEqual(y[1:], -1, 'mseq(%d)'%(j), precision=6)

    def test_gmseq(self):
        # we only test until 16, as longer sequences are too slow!
        for j in range(2, 17):
            x = signal.gmseq(j)
            self.assertArrayEqual(np.abs(x), np.ones(len(x)), precision=6)
            x_fft = np.fft.fft(x)
            y = np.abs(np.fft.ifft(x_fft*x_fft.conj()))
            self.assertApproxEqual(y[0], len(x), precision=6)
            self.assertArrayEqual(y[1:], 0, 'gmseq(%d)'%(j), precision=6)

    def test_bb2pb2bb(self):
        x = signal.bb2pb(np.ones(1024), 18000, 27000, 108000)
        self.assertArrayEqual(x[108:-108], np.sqrt(2)*np.cos(2*np.pi*27000*signal.time(x,108000))[108:-108], precision=3)
        x = np.random.normal(0, 1, 1024) + 1j*np.random.normal(0, 1, 1024)
        y = signal.bb2pb(x,  18000, 27000, 108000)
        z = signal.pb2bb(y, 108000, 27000,  18000)
        d = z[18:-18]-x[18:-18]
        self.assertLess(10*np.log10(np.mean(d*np.conj(d))), -25)
        self.assertArrayEqual(d, np.zeros_like(d), precision=1)

    def test_mfilter(self):
        x = np.random.normal(0, 1, 1000)
        y = signal.mfilter(x, np.pad(x, 10, 'constant'))
        self.assertEqual(len(y), 1020)
        self.assertEqual(np.argmax(y), 10)
        self.assertLess(np.max(y[:10]), np.max(y)/8)
        self.assertLess(np.max(y[11:]), np.max(y)/8)

    def test_lfilter0(self):
        x = np.random.normal(0, 1, 1000)
        hb = np.array([0, 0, 1, 0], dtype=np.float)
        self.assertArrayEqual(x, signal.lfilter0(hb, 1, x))

    def test_lfilter_gen(self):
        x = np.random.normal(0, 1, 1000)
        hb = np.array([0, 0, 1, 0], dtype=np.float)
        f = signal.lfilter_gen(hb, 1)
        y = [f.send(v) for v in x]
        self.assertArrayEqual(np.append([0, 0], x[:-2]), y)
        hb, ha = sp.iirfilter(4, 0.01, btype='lowpass')
        y1 = sp.lfilter(hb, ha, x)
        f = signal.lfilter_gen(hb, ha)
        y2 = [f.send(v) for v in x]
        self.assertArrayEqual(y1, y2, precision=6)

    def test_nco(self):
        nco = signal.nco_gen(27000, 108000, func=np.sin)
        x = [nco.__next__() for i in range(12)]
        x = np.append(x, nco.send(54000))
        x = np.append(x, [nco.__next__() for i in range(4)])
        self.assertArrayEqual(x, [0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1, 1, -1, 1, -1, 1], precision=6)
        fc = np.append([27000]*12, [54000]*5)
        x = signal.nco(fc, 108000, func=np.sin)
        self.assertArrayEqual(x, [0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1, 1, -1, 1, -1, 1], precision=6)

    def test_correlate_periodic(self):
        x = signal.gmseq(8)
        x_fft = np.fft.fft(x)
        y = np.fft.ifft(x_fft*x_fft.conj())
        z = signal.correlate_periodic(x)
        self.assertArrayEqual(y, z, precision=6)
        y = signal.correlate_periodic(x, x)
        self.assertArrayEqual(y, z)

    def test_goertzel(self):
        x1 = signal.cw(64, 1, 512)
        x2 = signal.cw(32, 1, 512)
        g1 = signal.goertzel(64, x1, 512)
        g2 = signal.goertzel(64, x2, 512)
        self.assertApproxEqual(g1, 512/2)
        self.assertApproxEqual(g2, 0)
        g1 = signal.goertzel(32, x1, 512)
        g2 = signal.goertzel(32, x2, 512)
        self.assertApproxEqual(g1, 0)
        self.assertApproxEqual(g2, 512/2)
        x2 = np.append(x2, [0])
        g2 = signal.goertzel(32, x2, 512, True)
        self.assertEqual(g2.size, 513)
        self.assertApproxEqual(np.abs(g2[-1]), 512/2)

    def test_detect_impulses(self):
        nsamp = 1000
        true_ind_imp = np.array([10, 115, 641, 888])
        x = np.zeros((nsamp))
        x[true_ind_imp] = 1
        x += np.random.normal(0, 0.1, nsamp)
        ind_imp, _ = signal.detect_impulses(x, fs=100000, k=10, tdist=1e-3)
        self.assertArrayEqual(true_ind_imp, ind_imp)

class CommsTestSuite(MyTestCase):

    def test_random_data(self):
        x = comms.random_data(1000)
        self.assertEqual(len(x), 1000)
        self.assertEqual(np.min(x), 0)
        self.assertEqual(np.max(x), 1)
        x = comms.random_data((1000, 2), m=8)
        self.assertEqual(np.shape(x), (1000, 2))
        self.assertEqual(np.min(x), 0)
        self.assertEqual(np.max(x), 7)

    def test_gray_code(self):
        self.assertArrayEqual(comms.gray_code(2), [0, 1])
        self.assertArrayEqual(comms.gray_code(4), [0, 1, 3, 2])
        self.assertArrayEqual(comms.gray_code(8), [0, 1, 3, 2, 6, 7, 5, 4])

    def test_invert_map(self):
        self.assertArrayEqual(comms.invert_map(comms.gray_code(8)), [0, 1, 3, 2, 7, 6, 4, 5])

    def test_sym2bi2sym(self):
        self.assertArrayEqual(comms.sym2bi([1, 2, 7], 8), [0, 0, 1, 0, 1, 0, 1, 1, 1])
        self.assertArrayEqual(comms.bi2sym([1, 0, 1, 1, 0, 1, 0, 0], 4), [2, 3, 1, 0])
        x = comms.random_data(1000, 8)
        self.assertArrayEqual(comms.bi2sym(comms.sym2bi(x, 8), 8), x)

    def test_ook(self):
        x = comms.ook()
        self.assertArrayEqual(x, [0, np.sqrt(2)], precision=4)

    def test_pam(self):
        x = comms.pam(2)
        self.assertArrayEqual(x, [-1, 1], precision=4)
        x = comms.pam(4)
        self.assertEqual(len(x), 4)
        self.assertApproxEqual(np.mean(x), 0, precision=4)
        self.assertApproxEqual(np.std(x), 1, precision=4)
        x = comms.pam(2, centered=False)
        self.assertArrayEqual(x, [0, np.sqrt(2)], precision=4)

    def test_psk(self):
        x = comms.psk(2)
        self.assertArrayEqual(x, [1, -1], precision=4)
        x = comms.psk(4)
        self.assertArrayEqual(np.sqrt(2)*x, [1+1j, -1+1j, 1-1j, -1-1j], precision=4)
        x = comms.psk(4, gray=False)
        self.assertArrayEqual(np.sqrt(2)*x, [1+1j, -1+1j, -1-1j, 1-1j], precision=4)
        x = comms.psk(8)
        self.assertArrayEqual(np.abs(x), np.ones(8), precision=4)

    def test_qam(self):
        x = comms.psk(16)
        self.assertEqual(len(x), 16)
        self.assertApproxEqual(np.mean(x), 0, precision=4)
        self.assertApproxEqual(np.std(x), 1, precision=4)
        x = comms.psk(64)
        self.assertEqual(len(x), 64)
        self.assertApproxEqual(np.mean(x), 0, precision=4)
        self.assertApproxEqual(np.std(x), 1, precision=4)

    def test_fsk(self):
        x = comms.fsk(2, 4)
        self.assertEqual(x.shape, (2, 4))
        self.assertArrayEqual(x, np.array([[1, 1j, -1, -1j], [1, -1j, -1, 1j]]), precision=4)
        x = comms.fsk(2, 8)
        self.assertEqual(x.shape, (2, 8))
        self.assertArrayEqual(x, np.array([[1, 1j, -1, -1j, 1, 1j, -1, -1j], [1, -1j, -1, 1j, 1, -1j, -1, 1j]]), precision=4)
        x = comms.fsk(4, 8)
        self.assertEqual(x.shape, (4, 8))

    def test_modulation(self):
        x = comms.random_data(1000)
        y = comms.modulate(x, comms.psk())
        self.assertArrayEqual(np.abs(y), np.ones(1000), precision=4)
        z = comms.demodulate(y, comms.psk())
        self.assertArrayEqual(x, z)
        x = comms.random_data(1000, m=4)
        y = comms.modulate(x, comms.psk(4))
        self.assertArrayEqual(np.abs(y), np.ones(1000), precision=4)
        z = comms.demodulate(y, comms.psk(4))
        self.assertArrayEqual(x, z)
        y = comms.diff_encode(comms.modulate(x, comms.psk(4))) * 1j
        z = comms.demodulate(comms.diff_decode(y), comms.psk(4))
        self.assertArrayEqual(x, z)
        y = comms.modulate(comms.random_data(1000), comms.fsk(2, 4))
        self.assertEqual(len(y), 4000)
        self.assertArrayEqual(np.abs(y), np.ones(4000), precision=4)
        y = comms.modulate(x, comms.pam(m=4, centered=False))
        z = comms.demodulate(-y, comms.pam(m=4, centered=False))
        self.assertArrayEqual(x, z)

    def test_diff(self):
        self.assertArrayEqual(comms.diff_encode([1, 1, -1, -1, -1, 1]), [ 1,  1,  1, -1,  1,  -1, -1])
        x = [1, 1, -1, -1j, -1j, 1j, 1, -1, -1j, 1]
        self.assertArrayEqual(comms.diff_decode(comms.diff_encode(x)), x)
        self.assertArrayEqual(comms.diff_decode(comms.diff_encode(x)*1j), x)

    def test_awgn(self):
        x = np.zeros(10000)
        self.assertApproxEqual(20*np.log10(1/np.std(comms.awgn(x, 10))), 10, precision=0)
        x = np.random.normal(0,1,10000)
        self.assertApproxEqual(20*np.log10(1/np.std(comms.awgn(x, 20)-x)), 20, precision=0)
        x = np.random.normal(0,1,10000) + 1j*np.random.normal(0,1,10000)
        self.assertApproxEqual(20*np.log10(1/np.std(comms.awgn(x, 6)-x)), 6, precision=0)
        x = 10*np.random.normal(0,1,10000)
        self.assertApproxEqual(20*np.log10(10/np.std(comms.awgn(x, 6, measured=True)-x)), 6, precision=0)
        x = 10*np.random.normal(0,1,10000) + 10j*np.random.normal(0,1,10000)
        self.assertApproxEqual(20*np.log10(10*np.sqrt(2)/np.std(comms.awgn(x, 6, measured=True)-x)), 6, precision=0)

    def test_ser_ber(self):
        x = comms.random_data(1000, m=2)
        self.assertEqual(comms.ser(x, x), 0)
        self.assertEqual(comms.ber(x, x), 0)
        y = np.array(x)
        y[50:150] = 1-y[50:150]
        self.assertEqual(comms.ber(y, x), 0.1)
        x = comms.random_data(1000, m=4)
        self.assertEqual(comms.ser(x, x), 0)
        self.assertRaises(ValueError, comms.ber, x, x)
        self.assertEqual(comms.ber(x, x, m=4), 0)
        y = np.array(x)
        y[50:150] = y[50:150]^1
        y[75:175] = y[75:175]^2
        self.assertEqual(comms.ber(y, x, m=4), 0.1)

    def test_rcosfir(self):
        # validation against MATLAB rcosdesign() generated filters
        self.assertArrayEqual(comms.rcosfir(0.25, 4, 11),  [-0.0017, -0.0021,  0.0000,  0.0044,  0.0083,  0.0076, -0.0000, -0.0121, -0.0210,
                                                            -0.0181,  0.0000,  0.0264,  0.0447,  0.0379, -0.0000, -0.0553, -0.0959, -0.0848,
                                                             0.0000,  0.1499,  0.3240,  0.4632,  0.5164,  0.4632,  0.3240,  0.1499,  0.0000,
                                                            -0.0848, -0.0959, -0.0553, -0.0000,  0.0379,  0.0447,  0.0264,  0.0000, -0.0181,
                                                            -0.0210, -0.0121, -0.0000,  0.0076,  0.0083,  0.0044,  0.0000, -0.0021, -0.0017], precision=4)
        self.assertArrayEqual(comms.rrcosfir(0.25, 4, 11), [ 0.0046,  0.0014, -0.0038, -0.0057, -0.0015,  0.0064,  0.0106,  0.0050, -0.0091,
                                                            -0.0213, -0.0188,  0.0030,  0.0327,  0.0471,  0.0265, -0.0275, -0.0852, -0.0994,
                                                            -0.0321,  0.1189,  0.3109,  0.4716,  0.5342,  0.4716,  0.3109,  0.1189, -0.0321,
                                                            -0.0994, -0.0852, -0.0275,  0.0265,  0.0471,  0.0327,  0.0030, -0.0188, -0.0213,
                                                            -0.0091,  0.0050,  0.0106,  0.0064, -0.0015, -0.0057, -0.0038,  0.0014,  0.0046], precision=4)

    def test_updown_rrc_conversion(self):
        x = comms.upconvert(np.ones(1024), 6, fc=27000, fs=108000)
        self.assertArrayEqual(x[108:-108], np.sqrt(2./6)*np.cos(2*np.pi*27000*signal.time(x,108000))[108:-108], precision=3)
        x = np.random.normal(0, 1, 1024) + 1j*np.random.normal(0, 1, 1024)
        rrcp = comms.rrcosfir(0.25, 6)
        y = comms.upconvert(x,  6, fc=0.5, g=rrcp)
        z = comms.downconvert(y, 6, fc=0.5, g=rrcp)
        delay = int((len(z)-len(x))/2)
        d = z[delay:-delay]-x
        self.assertLess(10*np.log10(np.mean(d*np.conj(d))), -40)
        self.assertArrayEqual(d.real, np.zeros_like(d, dtype=np.float), precision=1)
        self.assertArrayEqual(d.imag, np.zeros_like(d, dtype=np.float), precision=1)

    def test_updown_rect_conversion(self):
        x = np.random.normal(0, 1, 1024) + 1j*np.random.normal(0, 1, 1024)
        y = comms.upconvert(x,  16, fc=0.5)
        z = comms.downconvert(y, 16, fc=0.5)
        d = z-x
        self.assertLess(10*np.log10(np.mean(d*np.conj(d))), -10)
        self.assertArrayEqual(d.real, np.zeros_like(d, dtype=np.float), precision=1)
        self.assertArrayEqual(d.imag, np.zeros_like(d, dtype=np.float), precision=1)

class BeamformerTestSuite(MyTestCase):

    def test_normalize(self):
        x = np.empty((10, 1024), dtype=np.complex)
        for i in range(10):
            x[i,:] = np.random.normal(0, 1, 1024)*2*i - 1j*np.random.normal(0, 1, 1024)*i + i + i*0.5j
        y = bf.normalize(x, unit_variance=False)
        self.assertArrayEqual(np.mean(y, axis=-1), np.zeros(10), precision=6)
        v = np.mean(np.var(x, axis=-1))
        self.assertArrayEqual(np.var(y, axis=-1), [0, v, v, v, v, v, v, v, v, v], precision=6)
        y = bf.normalize(x, unit_variance=True)
        self.assertArrayEqual(np.mean(y, axis=-1), np.zeros(10), precision=6)
        self.assertArrayEqual(np.var(y, axis=-1), [0, 1, 1, 1, 1, 1, 1, 1, 1, 1], precision=6)

    def test_stft(self):
        x = np.ones((5, 1024))
        y = bf.stft(x, 64)
        self.assertEqual(y.shape, (5, 16, 64))
        self.assertArrayEqual(y[:,:,0], np.sqrt(64)*np.ones((5, 16)))
        self.assertArrayEqual(y[:,:,1:], np.zeros((5, 16, 63)))
        y = bf.stft(x, 64, 32)
        self.assertEqual(y.shape, (5, 31, 64))
        self.assertArrayEqual(y[:,:,0], np.sqrt(64)*np.ones((5, 31)))
        self.assertArrayEqual(y[:,:,1:], np.zeros((5, 31, 63)))
        y = bf.stft(x, 64, window='hanning')
        self.assertEqual(y.shape, (5, 16, 64))
        self.assertArrayEqual(y[:,:,0], np.sqrt(16)*np.ones((5, 16)), precision=0)
        self.assertArrayEqual(y[0,0,:], np.fft.fft(sp.get_window('hanning', 64))/np.sqrt(64))

    def test_steering_plane_wave(self):
        x = bf.steering_plane_wave(np.linspace(0, 5, 11), 1, 0)
        self.assertArrayEqual(x, np.zeros((1, 11)))
        x = bf.steering_plane_wave(np.linspace(0, 5, 11), 1, [-np.pi/2, np.pi/4])
        self.assertEqual(x.shape, (2, 11))
        self.assertArrayEqual(x[0], -np.linspace(2.5, -2.5, 11))
        self.assertArrayEqual(x[1], -np.linspace(-2.5, 2.5, 11)/np.sqrt(2))
        pos = [[0.0, 0], [0.5, 0], [1.0, 0], [1.5, 0], [2.0, 0]]
        x = bf.steering_plane_wave(pos, 1, [[-np.pi/2, 0], [np.pi/4, 0]])
        self.assertEqual(x.shape, (2, 5))
        self.assertArrayEqual(x[0], -np.linspace(1, -1, 5))
        self.assertArrayEqual(x[1], -np.linspace(-1, 1, 5)/np.sqrt(2))
        pos = [[0, 0.0], [0, 0.5], [0, 1.0], [0, 1.5], [0, 2.0]]
        x = bf.steering_plane_wave(pos, 1, [[0, -np.pi/2], [0, np.pi/4]])
        self.assertEqual(x.shape, (2, 5))
        self.assertArrayEqual(x[0], -np.linspace(1, -1, 5))
        self.assertArrayEqual(x[1], -np.linspace(-1, 1, 5)/np.sqrt(2))
        pos = [[0.0, 0, 0], [0.5, 0, 0], [1.0, 0, 0], [1.5, 0, 0], [2.0, 0, 0]]
        x = bf.steering_plane_wave(pos, 1, [[np.pi, 0], [np.pi/4, 0]])
        self.assertEqual(x.shape, (2, 5))
        self.assertArrayEqual(x[0], -np.linspace(1, -1, 5))
        self.assertArrayEqual(x[1], -np.linspace(-1, 1, 5)/np.sqrt(2), precision=6)
        x = bf.steering_plane_wave(pos, 2, [[np.pi, 0], [np.pi/4, 0]])
        self.assertEqual(x.shape, (2, 5))
        self.assertArrayEqual(x[0], -np.linspace(1, -1, 5)/2)
        self.assertArrayEqual(x[1], -np.linspace(-1, 1, 5)/np.sqrt(2)/2, precision=6)

    def test_delay_and_sum(self):
        sd = bf.steering_plane_wave(np.linspace(0, 5, 11), 1500, np.linspace(-np.pi/2, np.pi/2, 181))
        z = signal.cw(1500, 1, 84850)                          # 1.5 kHz passband signal from -45 deg
        y = np.zeros((11, z.shape[0]))
        for i in range(11):
            y[i,20*i:-1] = z[:-20*i-1]
        x = bf.delay_and_sum(y, 84850, sd)
        self.assertEqual(x.shape, (181, z.shape[0]-2*int(np.rint(2.5/1500*84850))))
        self.assertEqual((x**2).sum(axis=-1).argmax(), 45)

    def test_bartlett(self):
        sd = bf.steering_plane_wave(np.linspace(0, 5, 11), 1500, np.linspace(-np.pi/2, np.pi/2, 181))
        x = bf.bartlett(np.ones(11), 1500, sd)
        self.assertEqual(x.shape, (181,))
        self.assertEqual(np.argmax(x), 90)
        x = bf.bartlett(np.ones(11), 1500, sd, shading='hanning')
        self.assertEqual(x.shape, (181,))
        self.assertEqual(np.argmax(x), 90)
        y = np.exp(-2j*np.pi*np.linspace(2.5, -2.5, 11)/np.sqrt(2))   # baseband signal from +45 deg
        x = bf.bartlett(y, 1500, sd)
        self.assertEqual(np.argmax(x), 135)
        z = signal.cw(1500, 1, 8485)                          # 1.5 kHz passband signal from -45 deg
        y = np.zeros((11, z.shape[0]))
        for i in range(11):
            y[i,2*i:-1] = z[:-2*i-1]
        y1 = signal.pb2bb(y, 8485, 1500, 1000)
        x = bf.bartlett(y1, 1500, sd)
        self.assertEqual(x.shape, (181,))
        self.assertEqual(np.argmax(x), 45)
        x = bf.broadband(y1, 1000, 4, sd, f0=1500)
        self.assertEqual(x.shape, (181, 250))
        self.assertEqual(np.argmax(x[:,10]), 45)
        x = bf.broadband(y, 8485, 256, sd)
        self.assertEqual(x.shape, (181, 33))
        self.assertEqual(np.argmax(x.mean(axis=-1)), 45)
        y1 = signal.pb2bb(y, 8485, 1250, 1000)
        x = bf.broadband(y1, 1000, 16, sd, f0=1250)
        self.assertEqual(np.argmax(x[:,10]), 45)

    def test_bartlett_beampattern(self):
        sd = bf.steering_plane_wave(np.linspace(0, 5, 11), 1500, np.linspace(-np.pi/2, np.pi/2, 181))
        x = bf.bartlett_beampattern(90, 1500, sd)
        self.assertEqual(x.shape, (181,))
        self.assertEqual(np.argmax(x), 90)
        self.assertApproxEqual(x[90], 1.0, precision=6)
        x = bf.bartlett_beampattern(135, 1500, sd)
        self.assertEqual(x.shape, (181,))
        self.assertEqual(np.argmax(x), 135)
        self.assertApproxEqual(x[135], 1.0, precision=6)

    def test_capon(self):
        sd = bf.steering_plane_wave(np.linspace(0, 5, 11), 1500, np.linspace(-np.pi/2, np.pi/2, 181))
        x = bf.capon(np.ones(11), 1500, sd)
        self.assertEqual(x.shape, (181,))
        self.assertEqual(np.argmax(x), 90)
        y = np.exp(-2j*np.pi*np.linspace(2.5, -2.5, 11)/np.sqrt(2))   # baseband signal from +45 deg
        x = bf.capon(y, 1500, sd)
        self.assertEqual(np.argmax(x), 135)
        z = signal.cw(1500, 1, 8485)                          # 1.5 kHz passband signal from -45 deg
        y = np.zeros((11, z.shape[0]))
        for i in range(11):
            y[i,2*i:-1] = z[:-2*i-1]
        y1 = signal.pb2bb(y, 8485, 1500, 1000)
        x = bf.capon(y1, 1500, sd)
        self.assertEqual(x.shape, (181,))
        self.assertEqual(np.argmax(x), 45)
        x = bf.broadband(y1, 1000, 4, sd, f0=1500, beamformer=bf.capon)
        self.assertEqual(x.shape, (181, 250))
        self.assertEqual(np.argmax(x[:,10]), 45)
        x = bf.broadband(y, 8485, 256, sd, beamformer=bf.capon)
        self.assertEqual(x.shape, (181, 33))
        print(x.mean(axis=-1))
        self.assertLess(np.abs(np.argmax(x.mean(axis=-1))-45), 2)    # seems to give a result off by 1 deg
        y1 = signal.pb2bb(y, 8485, 1250, 1000)
        x = bf.broadband(y1, 1000, 16, sd, f0=1250, beamformer=bf.capon)
        self.assertEqual(np.argmax(x[:,10]), 45)

    def test_music(self):
        sd = bf.steering_plane_wave(np.linspace(0, 5, 11), 1500, np.linspace(-np.pi/2, np.pi/2, 181))
        x = bf.music(np.ones(11), 1500, sd, nsignals=3)
        self.assertEqual(x.shape, (181,))
        self.assertEqual(np.argmax(x), 90)
        y = np.exp(-2j*np.pi*np.linspace(2.5, -2.5, 11)/np.sqrt(2))   # baseband signal from +45 deg
        x = bf.music(y, 1500, sd)
        self.assertEqual(np.argmax(x), 135)
        z = signal.cw(1500, 1, 8485)                          # 1.5 kHz passband signal from -45 deg
        y = np.zeros((11, z.shape[0]))
        for i in range(11):
            y[i,2*i:-1] = z[:-2*i-1]
        y1 = signal.pb2bb(y, 8485, 1500, 1000)
        x = bf.music(y1, 1500, sd)
        self.assertEqual(x.shape, (181,))
        self.assertEqual(np.argmax(x), 45)
        x = bf.broadband(y1, 1000, 4, sd, f0=1500, beamformer=bf.music)
        self.assertEqual(x.shape, (181, 250))
        self.assertEqual(np.argmax(x[:,10]), 45)
        x = bf.broadband(y, 8485, 256, sd, beamformer=bf.music, nsignals=1)
        self.assertEqual(x.shape, (181, 33))
        print(x.mean(axis=-1))
        self.assertLess(np.abs(np.argmax(x.mean(axis=-1))-45), 3)    # seems to give a result off by 2 deg
        x = bf.broadband(y, 8485, 256, sd, beamformer=bf.music, nsignals=3)
        self.assertEqual(x.shape, (181, 33))
        print(x.mean(axis=-1))
        self.assertEqual(np.argmax(x.mean(axis=-1)),45)
        y1 = signal.pb2bb(y, 8485, 1250, 1000)
        x = bf.broadband(y1, 1000, 16, sd, f0=1250, beamformer=bf.music)
        self.assertEqual(np.argmax(x[:,10]), 45)

if __name__ == '__main__':
    unittest.main()
