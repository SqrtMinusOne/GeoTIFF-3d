import os
import shutil
import unittest

import numpy as np

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

        self.assertEqual(lonmin, proc.min_lon)
        self.assertEqual(latmin, proc.min_lat)
        self.assertEqual(lonmax, proc.max_lon)
        self.assertEqual(latmax, proc.max_lat)

        r = proc.max_rad(lat, lon)
        self.assertGreater(r, 0)

        xlen, ylen = proc.get_dimensions()
        self.assertLessEqual(
            np.abs(xlen * ylen - proc.points_estimate(r, 1)), 1)

        min_val, max_val = proc.get_value_limits()
        self.assertLessEqual(min_val, max_val)
        self.assertEqual(min_val, proc.min_val)
        self.assertEqual(max_val, proc.max_val)

    def test_normalize(self):
        proc = GeoTIFFProcessor()
        proc.open_file(SAMPLE_NAME)
        lat, lon = proc.center
        r = proc.max_rad(lat, lon) / 2
        df = proc.extract_to_pandas(lat, lon, r)
        df = proc.calculate_normals(df, normalize=True)
        self.assertTrue(0 <= min(df['x']) <= max(df['x']) <= 1)
        self.assertTrue(0 <= min(df['y']) <= max(df['y']) <= 1)
        self.assertTrue(0 <= min(df['value']) <= max(df['value']) <= 1)

    def test_save(self):
        proc = GeoTIFFProcessor()
        proc.open_file(SAMPLE_NAME)
        lat, lon = proc.center
        r = proc.max_rad(lat, lon) / 2
        proc.save('_temp/temp.tif', lat, lon, r)

    def test_modify(self):
        proc = GeoTIFFProcessor()
        proc.open_file(SAMPLE_NAME)

        # Crop
        lat, lon = proc.center
        r = proc.max_rad(lat, lon) / 2
        data = proc.modify_data(lat, lon, r, 1)
        xlen_old, ylen_old = proc.get_dimensions()
        xlen, ylen = proc.get_dimensions(data)
        self.assertLessEqual(np.abs(xlen_old / 2 - xlen), 1)
        self.assertLessEqual(np.abs(ylen_old / 2 - ylen), 1)

        # Make smaller
        r = proc.max_rad(lat, lon)
        data = proc.modify_data(lat, lon, r, 0.5)
        xlen, ylen = proc.get_dimensions(data)
        self.assertLessEqual(np.abs(xlen_old / 2 - xlen), 1)
        self.assertLessEqual(np.abs(ylen_old / 2 - ylen), 1)

        # Make larger (interpolate)
        data = proc.modify_data(lat, lon, r, 2)
        xlen, ylen = proc.get_dimensions(data)
        self.assertLessEqual(np.abs(xlen_old * 2 - xlen), 1)
        self.assertLessEqual(np.abs(ylen_old * 2 - ylen), 1)

    def test_to_pandas(self):
        proc = GeoTIFFProcessor()
        proc.open_file(SAMPLE_NAME)
        lat, lon = proc.center
        r = proc.max_rad(lat, lon) / 2
        df = proc.extract_to_pandas(lat, lon, r)

        df = proc.calculate_normals(df)
        self.assertGreater(len(df), 0)

        points = proc.points_estimate(r)
        self.assertLessEqual(np.abs(len(df) - points), points / 100)

        memory = sum(df.memory_usage())
        self.assertLessEqual(
            np.abs(proc.df_size_estimate(r) - memory), memory / 100)
