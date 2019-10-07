import georasters as gr
import numpy as np
import pandas as pd
from matplotlib import colors
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import \
    FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtCore import pyqtSignal
from haversine import haversine, Unit
from skimage.transform import resize

from loading_wrapper import LoadingThread

__all__ = ['GeoTIFFProcessor']


class GeoTIFFProcessor:
    """Main class to manage GeoTIFF file"""
    class PreprocessThread(LoadingThread):
        """A thread to do
        - extraction
        - normalization
        - normals calculation
        """
        df_ready = pyqtSignal(object)

        def __init__(self, proc, normalize=False, parent=None, *args,
                     **kwargs):
            """
            :param proc: an instance of GeoTIFFProcessor
            :param normalize: toggle normalization
            :param parent: a parent for QThread
            :param *args: Passed to GeoTIFFProcessor.modify_data
            :param **kwargs: Passed to GeoTIFFProcessor.modify_data
            """
            super().__init__(parent)
            self.operation = 'Data processing'
            self.proc = proc
            self.args = args
            self.kwargs = kwargs
            self.df = None
            self.normalize = normalize

        def run(self, df=None):
            def get_normal(p1, p2, p3):
                p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)
                v1 = p3 - p1  # These vectors belong to a surface
                v2 = p2 - p1
                c = np.cross(v1, v2)  # Cross product
                c = c / np.linalg.norm(c)
                x, y, z = c
                return x, y, z

            # Extraction
            if df is None:
                self.proc = GeoTIFFProcessor(self.proc.modify_data(
                    *self.args, **self.kwargs))
                df = self.proc.extract_to_pandas()
                self.updateStatus.emit('Calulating normals')
                self.set_interval(len(df))

            if self.normalize:
                df['x'] = [self.proc.normalizeLon(lon) for lon in df['x']]
                df['y'] = [self.proc.normalizeLat(lat) for lat in df['y']]
                df['value'] = [self.proc.normalizeValue(val)
                               for val in df['value']]

            # Getting normals
            max_row = max(df['row'])
            max_col = max(df['col'])

            normals = {"n_x": [], "n_y": [], "n_z": []}

            for index, row, col, value, x, y in df.itertuples():
                self.check_percent(index)
                polygon, border = self.proc.get_polygon(
                    row, col, max_row, max_col, df)
                p1, p2, p3, p4 = polygon

                n_x, n_y, n_z = get_normal(p1, p2, p3)
                normals["n_x"].append(n_x)
                normals["n_y"].append(n_y)
                normals["n_z"].append(n_z)

            df = df.join(pd.DataFrame(normals))
            self.df = df
            self.df_ready.emit(self.df)
            self.loadingDone.emit()

    def __init__(self, data=None):
        """
        :param data: an instance of GeoRaster
        """
        self.data = data
        if data is None:
            self.borders = 0., 0., 0., 0.
            self.min_val = self.max_val = 0
        else:
            self.borders = self.get_borders(data)
            self.min_val, self.max_val = self.get_value_limits(data)
        self._points_num = 0
        self._center = 0, 0
        self.fig, self.ax = None, None
        self.canvas = None

    # ==================== STATE ====================
    @property
    def data_loaded(self):
        """Is data loaded"""
        return self.data is not None

    @property
    def data_plotted(self):
        """Was data plotted"""
        return self.fig is not None

    @property
    def min_lon(self):
        return self.borders[0]

    @property
    def max_lon(self):
        return self.borders[1]

    @property
    def min_lat(self):
        return self.borders[2]

    @property
    def max_lat(self):
        return self.borders[3]

    # ==================== DATA PROPERTIES ====================
    def get_borders(self, data_=None):
        """Calculate Georaster's borders

        WARNING: This works incorrently with exracted data
        :param data: GeoRaster
        """
        data = self.data if data_ is None else data_
        xmin, xsize, xrot, ymax, yrot, ysize = data.geot
        xlen, ylen = self.get_dimensions(data)
        xlen = (xlen - 1) * xsize
        ylen = (ylen - 1) * ysize
        xmax = xmin + xlen
        ymin = ymax + ylen
        return xmin, xmax, ymin, ymax

    def get_centered_borders(self, data, center):
        """Calculate Georaster's borders and align them to a center.
        It is a workaround for bug in GeoTIFFProcessor.get_borders

        :param data: GeoRaster
        :param center: (lon, lat)
        """
        min_lon, max_lon, min_lat, max_lat = self.get_borders(data)
        cnt_lon, cnt_lat = center
        lon_r = (max_lon - min_lon) / 2
        lat_r = (max_lat - min_lat) / 2
        return cnt_lon - lon_r, cnt_lon + lon_r, \
            cnt_lat - lat_r, cnt_lat + lat_r

    def get_dimensions(self, data=None):
        """Get number of elements for each axis

        :param data: GeoRaster
        """
        data = self.data if data is None else data
        xlen = data.raster.count(axis=0)[0] \
            + np.ma.count_masked(data.raster, axis=0)[0]
        ylen = data.raster.count(axis=1)[0] \
            + np.ma.count_masked(data.raster, axis=1)[0]
        return xlen, ylen

    def get_value_limits(self, data=None):
        """Get minumum and maximum value (altitude)

        :param data: GeoRaster
        """
        data = self.data if data is None else data
        return data.raster.min(), data.raster.max()

    def get_real_scaling(self, data=None):
        """Get a real scale of altitude to longitude

        :param data: GeoRaster
        """
        data = self.data if data is None else data
        min_lon, max_lon, min_lat, max_lat = self.get_borders(data)
        min_val, max_val = self.get_value_limits(data)
        point_1 = (min_lat, min_lon)
        point_2 = (max_lat, min_lon)
        distance = haversine(point_1, point_2, unit=Unit.METERS)
        if distance != 0:
            return (max_val - min_val) / distance
        else:
            return 1

    def max_rad(self, lat, lon):
        """Get maximum radius to which can be extracted at given coordinates

        :param lat:
        :param lon:
        """
        # lonmin, lonmax, latmin, latmax = self.get_centered_borders(
        #     self.data, (lon, lat))
        lonmin, lonmax, latmin, latmax = self.get_borders()
        radmax = min(abs(lat - latmin), abs(lon - lonmin), abs(lat - latmax),
                     abs(lon - lonmax))
        return radmax

    @property
    def center(self):
        return self.get_center()

    def get_center(self, data=None):
        data = self.data if data is None else data
        lonmin, lonmax, latmin, latmax = self.get_borders(data)
        return (latmax - latmin) / 2 + latmin, \
            (lonmax - lonmin) / 2 + lonmin

    def points_estimate(self, r=None, coef=1):
        """Roughly estimate a number of points after extraction with given
        radius

        :param r: radius (must be in the same scale as lat and lon)
        :param coef: scale coefficient
        """
        if self._points_num == 0:
            xlen, ylen = self.get_dimensions()
            self._points_num = xlen * ylen
        if self._center == (0, 0):
            self._center = self.center
        points = self._points_num
        if r is not None:
            max_rad = self.max_rad(*self._center)
            if r < max_rad:
                points *= (r / self.max_rad(*self._center))**2
        points *= coef
        points = int(np.ceil(points))
        return points

    def df_size_estimate(self, *args, **kwargs):
        """Estimate size of a DataFrame with calculated normals after
        an extraction

        :param *args: Passed to GeoTIFFProcessor.points_estimate
        :param **kwargs: Passed to GeoTIFFProcessor.points_estimate
        """
        return self.points_estimate(*args, **kwargs) * 58 + 80

    def get_value(self, x, y, data=None):  # TODO Is this required?
        """Bilinear interpolation of a value at given coordinates

        :param x: longitude
        :param y: latitude
        :param data: GeoRaster
        """
        data = data if data is not None else self.data
        xlen, ylen = self.get_dimensions(data)
        x_1, x_2 = int(np.floor(x)), int(np.ceil(x))
        y_1, y_2 = int(np.floor(y)), int(np.ceil(y))
        x_2 = x_2 if x_2 < xlen else x_1
        y_2 = y_2 if y_2 < ylen else y_1
        v_11 = data.raster[x_1, y_1]
        v_12 = data.raster[x_1, y_2]
        v_21 = data.raster[x_2, y_1]
        v_22 = data.raster[x_2, y_2]
        if x_2 != x_1:
            v_r1 = (x_2 - x) / (x_2 - x_1) * v_11 + (x - x_1) / (x_2 -
                                                                 x_1) * v_21
            v_r2 = (x_2 - x) / (x_2 - x_1) * v_12 + (x - x_1) / (x_2 -
                                                                 x_1) * v_22
        else:
            v_r1, v_r2 = v_11, v_12
        if y_2 != y_1:
            v = (y_2 - y) / (y_2 - y_1) * v_r1 + (y - y_1) / (y_2 - y_1) * v_r2
        else:
            v = v_r1
        return v

    # ==================== NORMALIZATION ====================
    def normalizeLat(self, lat):
        """Returns a normalized (0 <= return <= 1) value of the latitude

        :param lat: latitude
        """
        if self.data.y_cell_size < 0:
            return (self.max_lat - lat) / (self.max_lat - self.min_lat)
        else:
            return (lat - self.min_lat) / (self.max_lat - self.min_lat)

    def normalizeLon(self, lon):
        """The same as GeoTIFFProcessor.normalizeLat

        :param lon: longitude
        """
        if self.data.x_cell_size < 0:
            return (self.max_lon - lon) / (self.min_lon - self.max_lon)
        else:
            return (lon - self.min_lon) / (self.max_lon - self.min_lon)

    def normalizeValue(self, value):
        """See GeoTIFFProcessor.normalizeLat

        :param value: altitude
        """
        return (value - self.min_val) / (self.max_val - self.min_val)

    def denormalizeValue(self, value):
        """Get an actual altitude by a normalized value

        :param value:
        """
        return value * (self.max_val - self.min_val) + self.min_val

    def normalizePoint(self, point):
        """Normalize (lon, lat, value)

        :param point:
        """
        lon, lat, value = point
        lon_ = self.normalizeLon(lon)
        lat_ = self.normalizeLat(lat)
        value_ = self.normalizeValue(value)
        return (lon_, lat_, value_)

    def normalizePoints(self, polygon):
        """Apply GeoTIFFProcessor.normalizePoints for each element

        :param polygon: iterable of (lon, lat, value)
        """
        return [self.normalizePoint(point) for point in polygon]

    # ==================== DATA PROCESSING & PANDAS ====================
    def modify_data(self, lat=None, lon=None, r=None, coef=1):
        """Extract a GeoRaster

        :param lat: latitude of center
        :param lon: longitude of center
        :param r: radius for extraction
        :param coef: if <1, data resolution will decrease
        """
        data = self.data
        if lat and lon and r:
            data = self.data.extract(lon, lat, r)
        if coef != 1:
            data = self._resize_data(data, coef)
        return data

    def _resize_data(self, data, coef):
        """Fix of georasters.resize for correct management of incomplete
        GeoTIFF-s

        :param data: GeoRaster
        :param coef: coefficient > 0 for resizing
        """
        xlen, ylen = self.get_dimensions(data)
        new_shape = (int(xlen * coef), int(ylen * coef))
        order = 0 if coef <= 1 else 1
        raster2 = data.raster.copy()
        raster2 = raster2.astype(float)
        raster2[data.raster.mask] = np.nan
        raster2 = resize(raster2, new_shape, order=order, mode='constant',
                         cval=False)
        raster2 = np.ma.masked_array(raster2, mask=np.isnan(raster2),
                                     fill_value=data.raster.fill_value)
        raster2 = raster2.astype(int)
        raster2[raster2.mask] = data.nodata_value
        raster2.mask = np.logical_or(np.isnan(raster2.data),
                                     raster2.data == data.nodata_value)
        geot = list(data.geot)
        [geot[-1], geot[1]] = np.array([geot[-1], geot[1]]) \
            * data.shape / new_shape
        return gr.GeoRaster(raster2, tuple(geot),
                            nodata_value=data.nodata_value,
                            projection=data.projection,
                            datatype=data.datatype)

    def _contour_cmap(self):
        """A red-green colormap for matplotlib"""
        cdict = {
            'red': [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)],
            'green': [(0.0, 1.0, 1.0), (1.0, 0.0, 0.0)],
            'blue': [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
        }
        return colors.LinearSegmentedColormap('rg', cdict, N=256)

    def get_contour(self, data=None, plot=False, *args, **kwargs):
        """Calculate a list of contour lines for GeoRaster

        :param data: GeoRaster
        :param plot: if True, a plot on GeoRaster.ax will be performed
        :param *args: passed to GeoTIFFProcessor.modify_data
        :param **kwargs: passed to GeoTIFFProcessor.modify_data

        :return: a list like [altitude, [(x1, y1), ...]]
        """
        def get_lon(x):
            return x_start + x * data.x_cell_size

        def get_lat(y):
            return y_start + y * data.y_cell_size

        data = self.data if data is None else data

        xlen, ylen = self.get_dimensions(data)
        lonmin, lonmax, latmin, latmax = self.get_borders(data)
        x_start = lonmin if data.x_cell_size > 0 else lonmax
        y_start = latmin if data.y_cell_size > 0 else latmax
        X = [i for i in range(0, xlen)] if data.x_cell_size > 0 \
            else [i for i in range(xlen - 1, -1, -1)]
        Y = [i for i in range(0, ylen)] if data.y_cell_size > 0 \
            else [i for i in range(ylen - 1, -1, -1)]
        X, Y = np.meshgrid(X, Y)
        Z = data.raster[Y, X]
        if plot:
            contour = self.ax.contour(X,
                                      Y,
                                      Z,
                                      cmap=self._contour_cmap(),
                                      *args,
                                      **kwargs)
            xticks = [i for i in np.linspace(0, xlen - 1, 10, dtype=int)]
            yticks = [i for i in np.linspace(0, ylen - 1, 10, dtype=int)]
            if data.x_cell_size < 0:
                xticks.reverse()
            if data.y_cell_size < 0:
                yticks.reverse()
            xtick_labels = [f"{get_lon(i):.2f}" for i in xticks]
            ytick_labels = [f"{get_lat(i):.2f}" for i in yticks]
            self.ax.invert_yaxis()
            self.ax.set_xticks(xticks)
            self.ax.set_xticklabels(xtick_labels, rotation=90)
            self.ax.set_yticks(yticks)
            self.ax.set_yticklabels(ytick_labels)
        else:
            contour = plt.contour(X, Y, Z, *args, **kwargs)

        result = []
        for level, coll in zip(contour.levels, contour.collections):
            for seg in coll.get_segments():
                new_seg = [(get_lon(x), get_lat(y)) for x, y in seg]
                result.append((level, new_seg))
        return result

    def extract_to_pandas(self, *args, **kwargs):
        """Extraction of a GeoRaster to pandas DataFrame with columns like
        [row, col, value, x, y]

        Unlike georasters.to_pandas, this method fills masked values with zeros

        :param *args: passed to GeoTIFFProcessor.modify_data
        :param **kwargs: GeoTIFFProcessor.modify_data
        """
        data = self.modify_data(*args, **kwargs)
        min_x, max_x, min_y, max_y = self.get_borders(data)
        xlen, ylen = self.get_dimensions(data)
        x_range = np.linspace(min_x, max_x, xlen)
        y_range = np.linspace(min_y, max_y, ylen)
        if data.x_cell_size < 0:
            x_range = np.flip(x_range)
        if data.y_cell_size < 0:
            y_range = np.flip(y_range)
        row_range = [i for i in range(xlen)]
        col_range = [i for i in range(ylen)]
        raster_replaced = np.ma.filled(data.raster, 0)
        assert len(x_range) == len(y_range) == len(row_range) == len(col_range)
        data = []
        [
            data.extend(
                [
                    (
                        row,
                        col,
                        raster_replaced[row, col],
                        x_range[col],
                        y_range[row]
                    )
                    for col in col_range
                ]
            )
            for row in row_range
        ]
        df = pd.DataFrame(data)
        df.columns = 'row', 'col', 'value', 'x', 'y'
        return df

    def calculate_normals(self, df, normalize=False):
        """Run GeoTIFFProcessor.PreprocessThread for given df

        :param df: DataFrame from GeoTIFFProcessor.extract_to_pandas
        """
        thread = self.PreprocessThread(self, normalize=normalize)
        thread.run(df)
        thread.wait()
        return thread.df

    def get_polygon(self, row, col, max_row, max_col, df):
        def get_index(row, col, max_row):
            """Get an element index by row & col
            """
            return row * (max_row + 1) + col

        # Get adjacent points
        # A point (i, j) > a normal for a polygon (i, j), (i+1, j), (i+1, j+1),
        # (i,j+1)
        target_row = row + 1 if row != max_row else row - 1
        target_col = col + 1 if col != max_col else col - 1
        index0 = get_index(row, col, max_row)
        index1 = get_index(target_row, col, max_row)
        index2 = get_index(row, target_col, max_row)
        index3 = get_index(target_row, target_col, max_row)

        border = False
        if target_row < row or target_col < col:
            border = True

        indices = (index0, index1, index3, index2)
        points = []
        for index in indices:
            point = df.loc[index]
            points.append((point.x, point.y, point.value))
        return points, border

    def polygon_generator(self, df):
        """Iterate through polygons in DataFrame

        :param df: df from GeoTIFFProcessor.extract_to_pandas
        """
        max_row = max(df['row'])
        max_col = max(df['col'])
        for i, row, col, value, x, y, n_x, n_y, n_z in df.itertuples():
            polygon, border = self.get_polygon(row, col, max_row, max_col, df)
            if border:
                continue
            normal = (n_x, n_y, n_z)
            yield polygon, normal

    # ==================== FILES & MPL ====================
    def open_file(self, name):
        self.data = gr.from_file(name)
        self.borders = self.get_borders(self.data)
        self.min_val, self.max_val = self.get_value_limits(self.data)
        self._points_num = 0
        self._center = (0, 0)

    def save(self, name, *args, **kwargs):
        if self.data_loaded:
            data = self.modify_data(*args, **kwargs)
            data.to_tiff(name)

    def init_canvas(self):
        """Init matplotlib FigureCanvas
        This can be used to embed stuff in PyQt"""
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.fig)
        return self.canvas

    def draw_preview(self, data=None, *args, **kwargs):
        """Draw a preview of a data on GeoTIFFProcessor.ax

        :param data: GeoRaster
        :param *args: passed to GeoTIFFProcessor.modify_data
        :param **kwargs: passed to GeoTIFFProcessor.modify_data
        """
        self.ax.cla()
        data = self.modify_data(*args, **kwargs) if data is None else data
        data.plot(ax=self.ax)
        self.canvas.draw()
