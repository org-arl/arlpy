from .context import arlpy

import unittest

class UtilsTestSuite(unittest.TestCase):

  def test_dB_conversions(self):
    self.assertEqual(arlpy.utils.mag2db(10.0), 20.0)
    self.assertEqual(arlpy.utils.db2mag(20.0), 10.0)
    self.assertEqual(arlpy.utils.pow2db(100.0), 20.0)
    self.assertEqual(arlpy.utils.db2pow(20.0), 100.0)

if __name__ == '__main__':
  unittest.main()
