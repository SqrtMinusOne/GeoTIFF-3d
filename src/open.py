from PyQt5.QtWidgets import QDialog, QApplication, QFileDialog
from PyQt5.QtCore import pyqtSignal
import sys
import georasters as gr
import numpy as np

from ui_compiled.open_dialog import Ui_OpenDialog


class OpenDialog(QDialog, Ui_OpenDialog):
    params_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.openButton.clicked.connect(self.on_open)
        self.connectStuff()
        self.params_changed.connect(self.set_rad_control)
        self.params_changed.connect(self.on_params_changed)
        self.okButton.clicked.connect(self.on_ok_pressed)
        self.border = 0., 0., 0., 0.
        self.lon, self.lat, self.r = 0., 0., 0.

    def connectStuff(self):
        def slider_to_spin(spin):
            return lambda val: spin.setValue(val / 100)

        def spin_to_slider(slider):
            return lambda val: slider.setValue(int(val * 100))

        sliders = [self.latSlider, self.lonSlider, self.radSlider]
        spins = [self.latSpin, self.lonSpin, self.radSpin]
        self.activate = sliders + spins

        self.latSpin.valueChanged.connect(lambda v: setattr(self, 'lat', v))
        self.lonSpin.valueChanged.connect(lambda v: setattr(self, 'lon', v))
        self.radSpin.valueChanged.connect(lambda v: setattr(self, 'r', v))

        for slider, spin in zip(sliders, spins):
            slider.valueChanged.connect(slider_to_spin(spin))
            spin.valueChanged.connect(spin_to_slider(slider))
            slider.valueChanged.connect(self.params_changed)
            spin.valueChanged.connect(self.params_changed)

    def on_open(self):
        name, filter_ = QFileDialog.getOpenFileName(self, 'Открыть файл',
                                                    filter='*.tif')
        if len(name) > 0:
            [item.setEnabled(True) for item in self.activate]
            self.open(name)

    def open(self, name):
        self.data = gr.from_file(name)
        self.borders = self.get_borders(self.data)
        self.set_controls()

    def get_borders(self, data):
        xmin, xsize, xrot, ymax, yrot, ysize = data.geot
        self.xsize, self.ysize = xsize, ysize
        xlen = data.count(axis=0)[0] * xsize
        ylen = data.count(axis=1)[0] * ysize
        xmax = xmin + xlen
        ymin = ymax + ylen
        return xmin, xmax, ymin, ymax

    def set_controls(self):
        lonmin, lonmax, latmin, latmax = self.borders
        lonmin_, latmin_ = int(np.ceil(lonmin)) * 100, int(np.ceil(latmin)) * 100
        lonmax_, latmax_ = int(lonmax) * 100, int(latmax) * 100
        self.latSlider.setRange(latmin_, latmax_)
        self.lonSlider.setRange(lonmin_, lonmax_)
        self.latSpin.setRange(latmin, latmax)
        self.lonSpin.setRange(lonmin, lonmax)
        self.latSpin.setValue((latmax - latmin) / 2 + latmin)
        self.lonSpin.setValue((lonmax - lonmin) / 2 + lonmin)
        self.minLat.setText(f"{latmin:.6f}")
        self.maxLat.setText(f"{latmax:.6f}")
        self.minLon.setText(f"{lonmin:.6f}")
        self.maxLon.setText(f"{lonmax:.6f}")

    def set_rad_control(self):
        lonmin, lonmax, latmin, latmax = self.borders
        radmax = min(abs(self.lat - latmin), abs(self.lon - lonmin),
                     abs(self.lat - latmax), abs(self.lon - lonmax))
        self.radSlider.setRange(0, int(radmax * 100))
        self.radSpin.setRange(0, radmax)

    def on_params_changed(self):
        # TODO Примерно оперативки
        points = (self.r * 2) ** 2 / abs((self.xsize * self.ysize))
        self.pointNumber.setText(str(int(np.round(points))))

    def on_ok_pressed(self):
        data = self.data.extract(self.lat, self.lon, self.r)
        data_pd = data.to_pandas()
        print(data_pd)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    dialog = OpenDialog()
    dialog.show()

    sys.exit(app.exec_())
