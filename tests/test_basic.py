from .context import arlpy

import unittest
import numpy as np
import scipy.signal

class MyTestCase(unittest.TestCase):

    def assertApproxEqual(self, x, y, precision=0):
        self.assertEqual(np.round(x, decimals=precision), np.round(y, decimals=precision))

    def assertArrayEqual(self, a, b, msg='', precision=None):
        if precision is None:
            np.testing.assert_array_equal(a, b, err_msg=msg)
        else:
            np.testing.assert_allclose(a, b, rtol=0, atol=np.power(10.0, -precision), err_msg=msg)

class UtilsTestSuite(MyTestCase):

    def test_dB_conversions(self):
        self.assertEqual(arlpy.utils.mag2db(10.0), 20.0)
        self.assertEqual(arlpy.utils.db2mag(20.0), 10.0)
        self.assertEqual(arlpy.utils.pow2db(100.0), 20.0)
        self.assertEqual(arlpy.utils.db2pow(20.0), 100.0)

class GeoTestSuite(MyTestCase):

    def test_pos(self):
        self.assertEqual(map(round, arlpy.geo.pos([1, 103, 20])), [277438, 110598, 20])
        self.assertEqual(arlpy.geo.zone([1, 103]), (48, 'N'))
        self.assertEqual(arlpy.geo.zone([1, 103, 20]), (48, 'N'))
        x = (1.25, 103.5, 10.0)
        y = arlpy.geo.latlong(arlpy.geo.pos(x), arlpy.geo.zone(x))
        self.assertEqual(tuple(np.round(y, 5)), x)
        self.assertEqual(arlpy.geo.d(x), (1.25, 103.5))
        self.assertEqual(arlpy.geo.dm(x), (1.0, 15.0, 103.0, 30.0))
        self.assertEqual(arlpy.geo.dms(x), (1.0, 15.0, 0.0, 103.0, 30.0, 0))
        self.assertEqual(arlpy.geo.dz(x), (1.25, 103.5, 10.0))
        self.assertEqual(arlpy.geo.dmz(x), (1.0, 15.0, 103.0, 30.0, 10.0))
        self.assertEqual(arlpy.geo.dmsz(x), (1.0, 15.0, 0.0, 103.0, 30.0, 0.0, 10.0))
        p1 = [100.0, 200.0, -5.0]
        p2 = [400.0, 600.0, -5.0]
        self.assertEqual(arlpy.geo.distance(p1, p1), 0.0)
        self.assertEqual(arlpy.geo.distance(p1, p2), 500.0)

class UwaTestSuite(MyTestCase):

    def test_soundspeed(self):
        self.assertApproxEqual(arlpy.uwa.soundspeed(27, 35, 10), 1539)

    def test_absorption(self):
        self.assertApproxEqual(arlpy.utils.mag2db(arlpy.uwa.absorption(50000)), -8)
        self.assertApproxEqual(arlpy.utils.mag2db(arlpy.uwa.absorption(100000)), -28)

    def test_absorption_filter(self):
        b = arlpy.uwa.absorption_filter(200000)
        w, h = scipy.signal.freqz(b, 1, 4)
        h = 20*np.log10(np.abs(h))
        self.assertEqual(list(np.round(h)), [0.0, -2.0, -8.0, -17.0])

    def test_density(self):
        self.assertApproxEqual(arlpy.uwa.density(27, 35), 1023)

    def test_reflection(self):
        self.assertApproxEqual(arlpy.uwa.reflection_coeff(0, 1200, 1600, 0, 1023, 1540), 0.0986, precision=4)
        self.assertApproxEqual(arlpy.uwa.reflection_coeff(0.5, 1200.0, 1600.0, 0.2, 1023, 1540), 0.0855-0.1278j, precision=4)

    def test_doppler(self):
        self.assertEqual(arlpy.uwa.doppler(0, 50000), 50000)
        self.assertApproxEqual(arlpy.uwa.doppler(10, 50000), 50325)
        self.assertApproxEqual(arlpy.uwa.doppler(-10, 50000), 49675)

class SigProcTestSuite(MyTestCase):

    def test_time(self):
        self.assertArrayEqual(arlpy.signal.time(1000, 500), np.arange(1000)/500.0)
        self.assertArrayEqual(arlpy.signal.time(np.zeros(1000), 500), np.arange(1000)/500.0)

    def test_cw(self):
        self.assertArrayEqual(arlpy.signal.cw(10000, 0.1, 50000), np.sin(2*np.pi*10000*np.arange(5000, dtype=np.float)/50000), precision=6)
        self.assertArrayEqual(arlpy.signal.cw(10000, 0.1, 50000, ('tukey', 0.1)), scipy.signal.tukey(5000, 0.1)*np.sin(2*np.pi*10000*np.arange(5000, dtype=np.float)/50000), precision=2)

    def test_sweep(self):
        self.assertArrayEqual(arlpy.signal.sweep(5000, 10000, 0.1, 50000), scipy.signal.chirp(np.arange(5000, dtype=np.float)/50000, 5000, 0.1, 10000, 'linear'))
        self.assertArrayEqual(arlpy.signal.sweep(5000, 10000, 0.1, 50000, 'hyperbolic'), scipy.signal.chirp(np.arange(5000, dtype=np.float)/50000, 5000, 0.1, 10000, 'hyperbolic'))
        self.assertArrayEqual(arlpy.signal.sweep(5000, 10000, 0.1, 50000, window=('tukey', 0.1)), scipy.signal.tukey(5000, 0.1)*scipy.signal.chirp(np.arange(5000, dtype=np.float)/50000, 5000, 0.1, 10000), precision=2)

    def test_envelope(self):
        x = np.random.normal(0, 1, 1000)
        self.assertArrayEqual(arlpy.signal.envelope(x), np.abs(scipy.signal.hilbert(x)))

    def test_mseq(self):
        # we only test until 16, as longer sequences are too slow!
        for j in range(2, 17):
            x = arlpy.signal.mseq(j)
            self.assertArrayEqual(np.abs(x), np.ones(len(x)))
            x_fft = np.fft.fft(x)
            y = np.fft.ifft(x_fft*x_fft.conj()).real
            self.assertEqual(round(y[0]), len(x), 'mseq(%d)'%(j))
            self.assertTrue((np.round(y[2:])==round(y[1])).all(), 'mseq(%d)'%(j))

    def test_freqz(self):
        # no regression test, since this is a graphics utility function
        pass

    def test_bb2pb2bb(self):
        x = arlpy.signal.bb2pb(np.ones(1024), 18000, 27000, 108000)
        self.assertArrayEqual(x[108:-108], np.sqrt(2)*np.cos(2*np.pi*27000*arlpy.signal.time(x,108000))[108:-108], precision=3)
        x = np.random.normal(0, 1, 1024) + 1j*np.random.normal(0, 1, 1024)
        y = arlpy.signal.bb2pb(x,  18000, 27000, 108000)
        z = arlpy.signal.pb2bb(y, 108000, 27000,  18000)
        d = z[18:-18]-x[18:-18]
        self.assertLess(10*np.log10(np.mean(d*np.conj(d))), -25)
        self.assertArrayEqual(d, np.zeros_like(d), precision=1)

    def test_mfilter(self):
        x = np.random.normal(0, 1, 1000)
        y = arlpy.signal.mfilter(x, np.pad(x, 10, 'constant'))
        self.assertEqual(len(y), 1020)
        self.assertEqual(np.argmax(y), 10)
        self.assertLess(np.max(y[:10]), np.max(y)/8)
        self.assertLess(np.max(y[11:]), np.max(y)/8)

    def test_lfilter0(self):
        x = np.random.normal(0, 1, 1000)
        hb = np.array([0, 0, 1, 0], dtype=np.float)
        self.assertArrayEqual(x, arlpy.signal.lfilter0(hb, 1, x))

class CommsTestSuite(MyTestCase):

    def test_random_data(self):
        x = arlpy.comms.random_data(1000)
        self.assertEqual(len(x), 1000)
        self.assertEqual(np.min(x), 0)
        self.assertEqual(np.max(x), 1)
        x = arlpy.comms.random_data((1000, 2), M=8)
        self.assertEqual(np.shape(x), (1000, 2))
        self.assertEqual(np.min(x), 0)
        self.assertEqual(np.max(x), 7)

    def test_gray_code(self):
        self.assertArrayEqual(arlpy.comms.gray_code(2), [0, 1])
        self.assertArrayEqual(arlpy.comms.gray_code(4), [0, 1, 3, 2])
        self.assertArrayEqual(arlpy.comms.gray_code(8), [0, 1, 3, 2, 6, 7, 5, 4])

    def test_invert_map(self):
        self.assertArrayEqual(arlpy.comms.invert_map(arlpy.comms.gray_code(8)), [0, 1, 3, 2, 7, 6, 4, 5])

    def test_pam(self):
        x = arlpy.comms.pam(2)
        self.assertArrayEqual(x, [-1, 1], precision=4)
        x = arlpy.comms.pam(4)
        self.assertEqual(len(x), 4)
        self.assertApproxEqual(np.mean(x), 0, precision=4)
        self.assertApproxEqual(np.std(x), 1, precision=4)

    def test_psk(self):
        x = arlpy.comms.psk(2)
        self.assertArrayEqual(x, [1, -1], precision=4)
        x = arlpy.comms.psk(4)
        self.assertArrayEqual(np.sqrt(2)*x, [1+1j, -1+1j, 1-1j, -1-1j], precision=4)
        x = arlpy.comms.psk(4, gray=False)
        self.assertArrayEqual(np.sqrt(2)*x, [1+1j, -1+1j, -1-1j, 1-1j], precision=4)
        x = arlpy.comms.psk(8)
        self.assertArrayEqual(np.abs(x), np.ones(8), precision=4)

    def test_qam(self):
        x = arlpy.comms.psk(16)
        self.assertEqual(len(x), 16)
        self.assertApproxEqual(np.mean(x), 0, precision=4)
        self.assertApproxEqual(np.std(x), 1, precision=4)
        x = arlpy.comms.psk(64)
        self.assertEqual(len(x), 64)
        self.assertApproxEqual(np.mean(x), 0, precision=4)
        self.assertApproxEqual(np.std(x), 1, precision=4)

    def test_iqplot(self):
        # no regression test, since this is a graphics utility function
        pass

    def test_modulation(self):
        x = arlpy.comms.random_data(1000, M=4)
        y = arlpy.comms.modulate(x, arlpy.comms.psk(4))
        self.assertArrayEqual(np.abs(y), np.ones(1000), precision=4)
        z = arlpy.comms.demodulate(y, arlpy.comms.psk(4))
        self.assertArrayEqual(x, z)

    def test_awgn(self):
        x = np.zeros(10000)
        self.assertApproxEqual(20*np.log10(1/np.std(arlpy.comms.awgn(x, 10))), 10, precision=0)
        x = np.random.normal(0,1,10000)
        self.assertApproxEqual(20*np.log10(1/np.std(arlpy.comms.awgn(x, 20)-x)), 20, precision=0)
        x = np.random.normal(0,1,10000) + 1j*np.random.normal(0,1,10000)
        self.assertApproxEqual(20*np.log10(1/np.std(arlpy.comms.awgn(x, 6)-x)), 6, precision=0)
        x = 10*np.random.normal(0,1,10000)
        self.assertApproxEqual(20*np.log10(10/np.std(arlpy.comms.awgn(x, 6, measured=True)-x)), 6, precision=0)
        x = 10*np.random.normal(0,1,10000) + 10j*np.random.normal(0,1,10000)
        self.assertApproxEqual(20*np.log10(10*np.sqrt(2)/np.std(arlpy.comms.awgn(x, 6, measured=True)-x)), 6, precision=0)

if __name__ == '__main__':
    unittest.main()
