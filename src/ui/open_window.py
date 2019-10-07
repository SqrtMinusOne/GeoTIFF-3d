import os

import numpy as np
from matplotlib.backends.backend_qt5agg import \
    NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QFileDialog, QMainWindow

from api import GeoTIFFProcessor
from ui_compiled.open_window import Ui_OpenWindow

from .loading_dialog import LoadingWrapper
from .main_window import MainWindow

__all__ = ['OpenDialog']


def format_coords(coord):
    return f"{coord:.6f}"


class OpenDialog(QMainWindow, Ui_OpenWindow):
    params_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.splitter.setStretchFactor(0, 2)
        self.splitter.setStretchFactor(1, 4)
        self.isSquareLabel.setVisible(False)

        self.openButton.clicked.connect(self.on_open)
        self.previewButton.clicked.connect(self.on_preview)
        self.saveButton.clicked.connect(self.on_save)
        self.connect_controls()
        self.params_changed.connect(self.on_params_changed)
        self.okButton.clicked.connect(self.on_ok_pressed)
        self.processor = GeoTIFFProcessor()
        self.lon, self.lat, self.r = 0., 0., 0.
        self.coef = 1.
        self.is_square = True
        self.thread = None

    def connect_controls(self):
        """Connect all the controls in OpenDialog"""

        def slider_to_spin(spin):
            return lambda val: spin.setValue(val / 100)

        def spin_to_slider(slider):
            return lambda val: slider.setValue(int(val * 100))

        sliders = [self.latSlider, self.lonSlider, self.radSlider]
        spins = [self.latSpin, self.lonSpin, self.radSpin]
        self.activate = sliders + spins + [self.previewButton, self.okButton]

        self.latSpin.valueChanged.connect(lambda v: setattr(self, 'lat', v))
        self.lonSpin.valueChanged.connect(lambda v: setattr(self, 'lon', v))
        self.radSpin.valueChanged.connect(lambda v: setattr(self, 'r', v))
        self.resSpin.valueChanged.connect(lambda v: setattr(self, 'coef', v))

        for slider, spin in zip(sliders, spins):
            slider.valueChanged.connect(slider_to_spin(spin))
            spin.valueChanged.connect(spin_to_slider(slider))
            spin.valueChanged.connect(self.params_changed)
            slider.valueChanged.connect(self.params_changed)
        self.resSpin.valueChanged.connect(self.params_changed)

    def set_rad_control(self):
        radmax = self.processor.max_rad(self.lat, self.lon)
        self.radSlider.setRange(0, int(radmax * 100))
        self.radSpin.setRange(0, radmax)

    def proc_params(self):
        return {
            'lat': self.lat,
            'lon': self.lon,
            'r': self.r,
            'coef': self.coef
        }

    def on_open(self):
        """Execute on open file"""
        name, filter_ = QFileDialog.getOpenFileName(
            self, 'Открыть файл', os.path.expanduser('~'), filter='*.tif')
        if len(name) > 0:
            [item.setEnabled(True) for item in self.activate]
            self.processor.open_file(name)
            self.set_controls()
            self.fileNameEdit.setText(name)

    def set_controls(self):
        """Update controls in accordance to a loaded file"""
        lonmin, lonmax, latmin, latmax = self.processor.borders
        lonmin_, latmin_ = int(np.ceil(lonmin * 100)), int(
            np.ceil(latmin * 100))
        lonmax_, latmax_ = int(lonmax * 100), int(latmax * 100)
        self.latSlider.setRange(latmin_, latmax_)
        self.lonSlider.setRange(lonmin_, lonmax_)
        self.latSpin.setRange(latmin, latmax)
        self.lonSpin.setRange(lonmin, lonmax)
        self.latSpin.setValue((latmax - latmin) / 2 + latmin)
        self.lonSpin.setValue((lonmax - lonmin) / 2 + lonmin)
        self.update_ranges()

    def update_ranges(self, data=None):
        """Update ranges for an extracted data

        :param data: GeoRaster
        """
        data = self.processor.data if data is None else data
        min_lon, max_lon, min_lat, max_lat = \
            self.processor.get_centered_borders(data, (self.lon, self.lat))
        min_val, max_val = self.processor.get_value_limits(data)

        self.minLat.setText(format_coords(min_lat))
        self.maxLat.setText(format_coords(max_lat))
        self.minLon.setText(format_coords(min_lon))
        self.maxLon.setText(format_coords(max_lon))
        self.minAlt.setText(str(int(min_val)))
        self.maxAlt.setText(str(int(max_val)))

    def on_preview(self):
        if not self.processor.data_plotted:
            self.init_matplotlib()
        data = self.processor.modify_data(**self.proc_params())
        xlen, ylen = self.processor.get_dimensions(data)
        self.is_square = xlen == ylen
        self.isSquareLabel.setVisible(not self.is_square)
        self.update_ranges(data)
        self.processor.draw_preview(data=data)

    def init_matplotlib(self):
        self.canvas = self.processor.init_canvas()
        self.nav_toolbar = NavigationToolbar(self.canvas, self)
        self.previewLabel.deleteLater()
        self.previewLayout.removeWidget(self.previewLabel)
        self.previewLayout.addWidget(self.canvas)
        self.previewLayout.addWidget(self.nav_toolbar)

    def on_params_changed(self):
        """Executed whenever a parameter is changed
        """
        self.set_rad_control()
        points = self.processor.points_estimate(self.r, self.coef)
        self.pointNumber.setText(str(int(np.round(points))))

        ram = self.processor.df_size_estimate(self.r, self.coef)
        prefixes = ['B', 'KB', 'MB', 'GB', 'TB']
        prefix = 0
        while ram > 1024:
            ram /= 1024
            prefix += 1
        self.ramUsage.setText(f"{ram:.2f} {prefixes[prefix]}")

        if self.autoUpdateCheckBox.isChecked() and self.processor.data_loaded:
            self.on_preview()

        self.okButton.setEnabled(self.r != 0)

    def on_ok_pressed(self):
        self.on_preview()
        if self.thread is None and self.is_square:
            self.thread = self.processor.PreprocessThread(
                self.processor, True, self, **self.proc_params())
            self.loading = LoadingWrapper(self.thread)
            self.thread.df_ready.connect(self.on_normals_ready)
            self.loading.start()

    def on_normals_ready(self, df):
        self.thread = None
        pass_proc = GeoTIFFProcessor(
            self.processor.modify_data(**self.proc_params()))
        self.window = MainWindow(pass_proc, df, self)
        self.window.show()
        self.hide()

    def on_save(self):
        filename, filter_ = QFileDialog.getSaveFileName(
            self, 'Сохранить', os.path.expanduser('~'), filter='*.tif')
        if len(filename) > 0:
            self.processor.save(filename, **self.proc_params())
