import georasters as gr
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import \
    FigureCanvasQTAgg as FigureCanvas

__all__ = ['GeoTIFFProcessor']


class GeoTIFFProcessor:
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
        radmax = min(abs(lat - latmin), abs(lon - lonmin),
                     abs(lat - latmax), abs(lon - lonmax))
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
        df = data.to_pandas()
        df = self.calculate_normals(df)
        return df

    def calculate_normals(self, df):
        def get_normal(p1, p2, p3):
            p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)
            v1 = p3 - p1  # Эти векторы принадлежат плоскости
            v2 = p2 - p1
            c = np.cross(v2, v1)  # Векторное произведение
            c = c / np.linalg.norm(c)
            x, y, z = c
            return x, y, z

        # Получить индекс элемента по строке и столбцу
        def get_index(row, col):
            return row * (max_row + 1) + col

        max_row = max(df['row'])
        max_col = max(df['col'])

        # Хранилище нормалей
        normals = {"n_x": [], "n_y": [], "n_z": []}

        for index, row, col, value, x, y in df.itertuples():
            # Получить соседние точки
            target_row = row + 1 if row != max_row else row - 1
            target_col = col + 1 if col != max_col else col - 1
            index1 = get_index(target_row, col)
            index2 = get_index(row, target_col)

            p1 = x, y, value
            p2, p3 = df.loc[index1], df.loc[index2]
            p2 = p2.x, p2.y, p2.value
            p3 = p3.x, p3.y, p3.value

            # Посчитать нормаль
            n_x, n_y, n_z = get_normal(p1, p2, p3)
            normals["n_x"].append(n_x)
            normals["n_y"].append(n_y)
            normals["n_z"].append(n_z)

        df = df.join(pd.DataFrame(normals))
        return df
