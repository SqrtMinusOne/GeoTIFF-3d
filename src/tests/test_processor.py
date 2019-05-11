import unittest
import os
import shutil

from api import GeoTIFFProcessor


SAMPLE_NAME = '../samples/everest.tif'


class TestGeoTIFFProcessor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not os.path.exists('_temp'):
            os.mkdir('_temp')

    @classmethod
    def tearDownClass(cls):
        if os.path.exists('_temp'):
            shutil.rmtree('_temp')

    def test_read_file(self):
        proc = GeoTIFFProcessor()
        self.assertFalse(proc.data_loaded)
        self.assertFalse(proc.data_plotted)

        proc.open_file(SAMPLE_NAME)
        self.assertTrue(proc.data_loaded)

        lonmin, lonmax, latmin, latmax = proc.borders
        lat, lon = proc.center
        self.assertTrue(latmin < lat < latmax)
        self.assertTrue(lonmin < lon < lonmax)

        r = proc.max_rad(lat, lon)
        self.assertGreater(r, 0)

    def test_save(self):
        proc = GeoTIFFProcessor()
        proc.open_file(SAMPLE_NAME)
        lat, lon = proc.center
        r = proc.max_rad(lat, lon) / 2
        proc.save('_temp/temp.tif', lat, lon, r)

    def test_pandas(self):
        proc = GeoTIFFProcessor()
        proc.open_file(SAMPLE_NAME)
        lat, lon = proc.center
        r = proc.max_rad(lat, lon) / 2
        df = proc.extract_to_pandas(lat, lon, r)
        self.assertGreater(len(df), 0)
