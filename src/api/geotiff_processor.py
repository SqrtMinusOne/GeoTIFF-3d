import georasters as gr
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import \
    FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtCore import pyqtSignal

from loading_wrapper import LoadingThread

__all__ = ['GeoTIFFProcessor']


class GeoTIFFProcessor:
    class PreprocessThread(LoadingThread):
        df_ready = pyqtSignal(object)

        def __init__(self, proc, parent=None, *args, **kwargs):
            super().__init__(parent)
            self.operation = 'Data processing'
            self.proc = proc
            self.args = args
            self.kwargs = kwargs
            self.df = None

        def run(self, df=None):
            def get_normal(p1, p2, p3):
                p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)
                v1 = p3 - p1  # Эти векторы принадлежат плоскости
                v2 = p2 - p1
                c = np.cross(v2, v1)  # Векторное произведение
                c = c / np.linalg.norm(c)
                x, y, z = c
                return x, y, z

            # Extraction
            if df is None:
                df = self.proc.extract_to_pandas(*self.args, **self.kwargs)

                self.updateStatus.emit('Calulating normals')
                self.set_interval(len(df))

            # Getting normals
            max_row = max(df['row'])
            max_col = max(df['col'])

            # Хранилище нормалей
            normals = {"n_x": [], "n_y": [], "n_z": []}

            for index, row, col, value, x, y in df.itertuples():
                self.check_percent(index)
                polygon, border = self.proc.get_polygon(
                    row, col, max_row, max_col, df)
                p1, p2, p3, p4 = polygon
                # Посчитать нормаль
                n_x, n_y, n_z = get_normal(p1, p2, p3)
                normals["n_x"].append(n_x)
                normals["n_y"].append(n_y)
                normals["n_z"].append(n_z)

            df = df.join(pd.DataFrame(normals))
            self.df = df
            self.df_ready.emit(self.df)
            self.loadingDone.emit()

    def __init__(self, data=None):
        self.data = data
        if data is None:
            self.border = 0., 0., 0., 0.
            self.xsize, self.ysize = 0, 0
        else:
            self.borders = self.get_borders(data)
        self.fig, self.ax = None, None
        self.canvas = None

    # ==================== STATE ====================
    @property
    def data_loaded(self):
        return self.data is not None

    @property
    def data_plotted(self):
        return self.fig is not None

    # ==================== DATA PROPERTIES ====================
    def get_borders(self, data_=None):
        """Вычислить границы для GeoRaster'a

        :param data: GeoRaster
        """
        data = self.data if data_ is None else data_
        xmin, xsize, xrot, ymax, yrot, ysize = data.geot
        self.xsize, self.ysize = xsize, ysize
        xlen = data.count(axis=0)[0] * xsize
        ylen = data.count(axis=1)[0] * ysize
        xmax = xmin + xlen
        ymin = ymax + ylen
        return xmin, xmax, ymin, ymax

    def get_dimensions(self, data=None):
        data = self.data if data is None else data
        xlen = data.count(axis=0)[0]
        ylen = data.count(axis=1)[0]
        return xlen, ylen

    def max_rad(self, lat, lon):
        lonmin, lonmax, latmin, latmax = self.borders
        radmax = min(abs(lat - latmin), abs(lon - lonmin), abs(lat - latmax),
                     abs(lon - lonmax))
        return radmax

    @property
    def center(self):
        lonmin, lonmax, latmin, latmax = self.borders
        return (latmax - latmin) / 2 + latmin, \
            (lonmax - lonmin) / 2 + lonmin

    def points_estimate(self, r, coef=1):
        xlen, ylen = self.get_dimensions()
        points = xlen * ylen
        points *= (r / self.max_rad(*self.center))**2
        points *= coef
        points = int(np.ceil(points))
        return points

    def df_size_estimate(self, *args, **kwargs):
        return self.points_estimate(*args, **kwargs) * 58 + 80

    def get_value(self, x, y, data=None):  # TODO Is this required?
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

    # ==================== DATA PROCESSING & PANDAS ====================
    def _modify_data(self, lat=None, lon=None, r=None, coef=1):
        data = self.data
        if lat and lon and r:
            data = self.data.extract(lon, lat, r)
        if coef != 1:
            xlen, ylen = self.get_dimensions(data)
            new_shape = (int(xlen * coef), int(ylen * coef))
            data = data.resize(new_shape, cval=True)
        return data

    def get_contour(self, data=None, plot=False, *args, **kwargs):
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
            contour = self.ax.contour(X, Y, Z, *args, **kwargs)
        else:
            contour = plt.contour(X, Y, Z, *args, **kwargs)

        result = []
        for level, coll in zip(contour.levels, contour.collections):
            for seg in coll.get_segments():
                new_seg = [(get_lon(x), get_lat(y)) for x, y in seg]
                result.append((level, new_seg))
        return result

    def extract_to_pandas(self, *args, **kwargs):
        data = self._modify_data(*args, **kwargs)
        return data.to_pandas()

    def calculate_normals(self, df):
        thread = self.PreprocessThread(self)
        thread.run(df)
        thread.wait()
        return thread.df

    def get_polygon(self, row, col, max_row, max_col, df):
        # Получить индекс элемента по строке и столбцу
        def get_index(row, col, max_row):
            return row * (max_row + 1) + col

        # Получить соседние точки
        # Точка (i, j) - нормаль для полигона (i, j), (i+1, j), (i+1, j+1),
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

    def save(self, name, *args, **kwargs):
        if self.data_loaded:
            data = self._modify_data(*args, **kwargs)
            data.to_tiff(name)

    def init_canvas(self):
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.fig)
        return self.canvas

    def draw_preview(self, *args, **kwargs):
        self.ax.cla()
        data = self._modify_data(*args, **kwargs)
        data.plot(ax=self.ax)
        self.canvas.draw()
