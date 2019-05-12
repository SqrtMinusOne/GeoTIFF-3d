from PyQt5.QtCore import pyqtSignal
import georasters as gr
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import \
    FigureCanvasQTAgg as FigureCanvas

from loading_wrapper import LoadingThread


__all__ = ['GeoTIFFProcessor']


class GeoTIFFProcessor:
    class PreprocessThread(LoadingThread):
        df_ready = pyqtSignal(object)

        def __init__(self, proc, lat, lon, r, parent=None):
            super().__init__(parent)
            self.operation = 'Data processing'
            self.proc = proc
            self.args = lat, lon, r
            self.df = None

        def run(self):
            def get_normal(p1, p2, p3):
                p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)
                v1 = p3 - p1  # Эти векторы принадлежат плоскости
                v2 = p2 - p1
                c = np.cross(v2, v1)  # Векторное произведение
                c = c / np.linalg.norm(c)
                x, y, z = c
                return x, y, z

            # Extraction
            df = self.proc.extract_to_pandas(*self.args)

            self.updateStatus.emit('Calulating normals')
            self.set_interval(len(df))

            # Getting normals
            max_row = max(df['row'])
            max_col = max(df['col'])

            # Хранилище нормалей
            normals = {"n_x": [], "n_y": [], "n_z": []}

            for index, row, col, value, x, y in df.itertuples():
                self.check_percent(index)
                polygon, border = self.proc.get_polygon(row, col, max_row,
                                                        max_col, df)
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

    def __init__(self):
        self.data = None
        self.fig, self.ax = None, None
        self.border = 0., 0., 0., 0.
        self.xsize, self.ysize = 0, 0
        self.canvas = None

    @property
    def data_loaded(self):
        return self.data is not None

    @property
    def data_plotted(self):
        return self.fig is not None

    def get_borders(self, data):
        """Вычислить границы для GeoRaster'a

        :param data: GeoRaster
        """
        xmin, xsize, xrot, ymax, yrot, ysize = data.geot
        self.xsize, self.ysize = xsize, ysize
        xlen = data.count(axis=0)[0] * xsize
        ylen = data.count(axis=1)[0] * ysize
        xmax = xmin + xlen
        ymin = ymax + ylen
        return xmin, xmax, ymin, ymax

    def open_file(self, name):
        self.data = gr.from_file(name)
        self.borders = self.get_borders(self.data)

    def save(self, name, lat, lon, r):
        if self.data_loaded:
            data = self.data.extract(lon, lat, r)
            data.to_tiff(name)

    def init_canvas(self):
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.fig)
        return self.canvas

    def draw_preview(self, lat, lon, r):
        self.ax.cla()
        data = self.data.extract(lon, lat, r)
        data.plot(ax=self.ax)
        self.canvas.draw()

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

    def points_estimate(self, r):
        points = (r * 2)**2 / abs((self.xsize * self.ysize))
        return points

    def extract_to_pandas(self, lat, lon, r):
        data = self.data.extract(lon, lat, r)
        return data.to_pandas()

    def calculate_normals(self, df):
        thread = self.PreprocessThread(self, df)
        thread.run()
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
