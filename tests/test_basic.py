from .context import arlpy

import unittest
import numpy as np
import scipy.signal

class UtilsTestSuite(unittest.TestCase):

    def test_dB_conversions(self):
        self.assertEqual(arlpy.utils.mag2db(10.0), 20.0)
        self.assertEqual(arlpy.utils.db2mag(20.0), 10.0)
        self.assertEqual(arlpy.utils.pow2db(100.0), 20.0)
        self.assertEqual(arlpy.utils.db2pow(20.0), 100.0)

class GeoTestSuite(unittest.TestCase):

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

class UwaTestSuite(unittest.TestCase):

    def assertApproxEqual(self, x, y, precision=0):
        self.assertEqual(np.round(x, decimals=precision), np.round(y, decimals=precision))

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

class SigProcTestSuite(unittest.TestCase):

    def assertArrayEqual(self, a, b, msg='', precision=None):
        if precision is None:
            np.testing.assert_array_equal(a, b, err_msg=msg)
        else:
            np.testing.assert_allclose(a, b, rtol=0, atol=np.power(10.0, -precision), err_msg=msg)

    def test_time(self):
        self.assertArrayEqual(arlpy.sigproc.time(1000, 500), np.arange(1000)/500.0)
        self.assertArrayEqual(arlpy.sigproc.time(np.zeros(1000), 500), np.arange(1000)/500.0)

    def test_cw(self):
        self.assertArrayEqual(arlpy.sigproc.cw(10000, 0.1, 50000), np.sin(2*np.pi*10000*np.arange(5000, dtype=np.float)/50000), precision=6)
        self.assertArrayEqual(arlpy.sigproc.cw(10000, 0.1, 50000, ('tukey', 0.1)), scipy.signal.tukey(5000, 0.1)*np.sin(2*np.pi*10000*np.arange(5000, dtype=np.float)/50000), precision=2)

    def test_envelope(self):
        x = np.random.normal(0, 1, 1000)
        self.assertArrayEqual(arlpy.sigproc.envelope(x), np.abs(scipy.signal.hilbert(x)))

    def test_mseq(self):
        # we only test until 16, as longer sequences are too slow!
        for j in range(2, 17):
            x = arlpy.sigproc.mseq(j)
            self.assertArrayEqual(np.abs(x), np.ones(len(x)))
            x_fft = np.fft.fft(x)
            y = np.fft.ifft(x_fft*x_fft.conj()).real
            self.assertEqual(round(y[0]), len(x), 'mseq(%d)'%(j))
            self.assertTrue((np.round(y[2:])==round(y[1])).all(), 'mseq(%d)'%(j))

    def test_freqz(self):
        # no test, since this is a graphics utility function
        pass

    def test_bb2pb2bb(self):
        x = arlpy.sigproc.bb2pb(np.ones(1024), 18000, 27000, 108000)
        self.assertArrayEqual(x[108:-108], np.sqrt(2)*np.cos(2*np.pi*27000*arlpy.sigproc.time(x,108000))[108:-108], precision=3)
        x = np.random.normal(0, 1, 1024) + 1j*np.random.normal(0, 1, 1024)
        y = arlpy.sigproc.bb2pb(x,  18000, 27000, 108000)
        z = arlpy.sigproc.pb2bb(y, 108000, 27000,  18000)
        d = z[18:-18]-x[18:-18]
        self.assertLess(10*np.log10(np.mean(d*np.conj(d))), -25)
        self.assertArrayEqual(d, np.zeros_like(d), precision=1)

    def test_mfilter(self):
        x = np.random.normal(0, 1, 1000)
        y = arlpy.sigproc.mfilter(x, np.pad(x, 10, 'constant'))
        self.assertEqual(len(y), 1020)
        self.assertEqual(np.argmax(y), 10)
        self.assertLess(np.max(y[:10]), np.max(y)/10)
        self.assertLess(np.max(y[11:]), np.max(y)/10)

    def test_lfilter0(self):
        x = np.random.normal(0, 1, 1000)
        hb = np.array([0, 0, 1, 0], dtype=np.float)
        self.assertArrayEqual(x, arlpy.sigproc.lfilter0(hb, 1, x))

if __name__ == '__main__':
    unittest.main()
