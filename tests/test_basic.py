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

if __name__ == '__main__':
    unittest.main()
