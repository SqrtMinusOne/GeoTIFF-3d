import numpy as np
from OpenGL import GL
from PyQt5.QtCore import QPoint, Qt
from PyQt5.QtGui import (QColor, QCursor, QMatrix4x4, QOpenGLShader,
                         QOpenGLShaderProgram, QVector3D, QVector4D)
from PyQt5.QtWidgets import QMainWindow

from ui.widgets import ElevationGraphWidget, MinimapGraphWidget
from ui_compiled.mainwindow import Ui_MainWindow
import os
import sys

__all__ = ['MainWindow']


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, processor, df, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.processor = processor
        # Data
        self.df = df

        # UI
        self.setupUi(self)
        self.setupControls()
        self.keyPressEvent = self.keyPressed
        self.mouseMoveEvent = self.mouseMoved

        # Control & Display
        self.mouse_grabbed = False

        self.camera_pos = QVector3D(0.5, 0.5, -2)
        self.center = QVector3D(0.5, 0, 0.5)
        self.rot_center = QVector3D(0.5, 0.5, 0.5)
        self.camera_rot = QVector3D(0, 0, 1)
        self.scale_vec = QVector3D(1, 1, 1)

        self.light_pos = QVector3D(self.xLightSpinBox.value(),
                                   self.yLightSpinBox.value(),
                                   self.zLightSpinBox.value())
        self.ambient = self.ambientSlider.value() / 100
        self.diffuse = self.diffuseSlider.value() / 100
        self.alpha = self.alphaSlider.value() / 100

        # Drawing
        self.normals = []
        self.colors = []
        self.coords_array = []
        self.update_light = False
        self.update_buffer = False

        self.show_grid = self.gridCheckBox.isChecked()
        self.show_contour = self.contourCheckBox.isChecked()
        self.contour_levels = self.contourLevelSpinBox.value()
        self.show_light_lines = True
        self.grid_freq = 10

        self.grid_color = QVector4D(1, 1, 1, 1)
        self.contour_color = QVector4D(1, 1, 1, 1)
        self.light_line_color = QVector4D(1, 0.6, 0, 1)

        self.prepareScene()
        self.updateUi()

        self.shaders = QOpenGLShaderProgram()
        self.openGLWidget.initializeGL = self.initializeGL
        self.openGLWidget.paintGL = self.paintGL

    def updateUi(self):
        self.splitDockWidget(self.displayDockWidget, self.elevationDockWidget,
                             Qt.Vertical)
        self.splitDockWidget(self.elevationDockWidget, self.displayDockWidget,
                             Qt.Vertical)

        self.tabifyDockWidget(self.elevationDockWidget, self.minimapDockWidget)
        self.tabifyDockWidget(self.displayDockWidget, self.lightDockWidget)
        self.tabifyDockWidget(self.projDockWidget, self.additionalDockWidget)
        self.tabifyDockWidget(self.elevationDockWidget, self.cameraDockWidget)
        self.lightDockWidget.raise_()
        self.additionalDockWidget.raise_()
        self.elevationDockWidget.raise_()

        self.elevationWidget = ElevationGraphWidget(
            self.processor.min_val,
            self.processor.max_val,
            self.processor.denormalizeValue(self.camera_pos.y()),
            width=240,
            height=100)
        self.minimapWidget = MinimapGraphWidget(self.processor,
                                                self.camera_pos,
                                                self.camera_rot,
                                                width=240,
                                                height=100)
        self.elevationWidgetLayout.addWidget(self.elevationWidget)
        self.minimapLayout.addWidget(self.minimapWidget)
        self.mapDockWidgetControls()

        self.actionOpenAnother.triggered.connect(self.onOpenAnother)

    def onOpenAnother(self):
        self.parent.show()
        self.hide()
        self.deleteLater()

    def mapDockWidgetControls(self):
        self.dock_widgets = [
            self.lightDockWidget, self.cameraDockWidget,
            self.additionalDockWidget, self.minimapDockWidget,
            self.displayDockWidget, self.projDockWidget,
            self.elevationDockWidget
        ]
        self.dock_actions = [
            self.actionShowLightSourceDW, self.actionShowCameraDW,
            self.actionShowAdditionalDW, self.actionShowMinimapDW,
            self.actionShowDisplayDW, self.actionShowProjectionDW,
            self.actionShowElevationDW
        ]
        for dock_widget, action in zip(self.dock_widgets, self.dock_actions):

            def wrapper(action):
                def dock_widget_close_event(event):
                    action.setChecked(False)
                    event.accept()

                return dock_widget_close_event

            dock_widget.closeEvent = wrapper(action)
            action.triggered.connect(dock_widget.setVisible)

    # ==================== PREPARATION ====================
    def prepareScene(self):
        self.coords_array = []
        self.colors = []
        self.normals = []
        polygons, normals, colors = self.getMapPolygons()
        self.map_data = self.preparePolygons(polygons, normals, colors)
        # self.normal_data = self.prepareNormalLines(polygons, normals, colors)

        polygons, normals, colors = self.getLightSourceCoords()
        self.light_data = self.preparePolygons(polygons, normals, colors)
        if self.show_light_lines:
            lines, line_colors = self.getLightLines()
            self.light_lines_data = self.prepareLines(lines, line_colors)

        if self.show_grid:
            lines, line_colors = self.getGrid()
            self.grid_data = self.prepareLines(lines, line_colors)
        if self.show_contour:
            self.contour_data = self.getContour()

    # POLYGONS
    def getLightSourceCoords(self):
        polygons = np.array([((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)),
                             ((0, 0, 1), (0, 1, 1), (1, 1, 1), (1, 0, 1)),
                             ((0, 0, 0), (0, 0, 1), (1, 0, 1), (1, 0, 0)),
                             ((0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)),
                             ((0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)),
                             ((1, 0, 0), (1, 0, 1), (1, 1, 1), (1, 1, 0))])
        normals = [(0, 0, -1), (0, 0, 1), (0, -1, 0), (0, 1, 0), (-1, 0, 0),
                   (1, 0, 0)]

        normals_vec = []
        [[
            normals_vec.append(QVector3D(*normals[i]))
            for _ in range(len(polygons[i]))
        ] for i in range(len(polygons))]

        center = np.array(
            (self.light_pos.x(), self.light_pos.y(), self.light_pos.z()))
        delta = np.array((0.5, 0.5, 0.5))
        polygons = [[(p - delta) * 0.05 + center for p in side]
                    for side in polygons]
        colors = []
        [
            colors.append(QVector4D(1, 153 / 255, 0, 1) * self.diffuse)
            for _ in range(len(normals_vec))
        ]
        return polygons, normals_vec, colors

    def getMapPolygons(self):
        polygons, normals, colors = [], [], []
        for polygon, normal in self.processor.polygon_generator(self.df):
            polygons.append(self.swapPoints(polygon))
            [
                normals.append(QVector3D(*self.swapPoint(*normal)))
                for _ in polygon
            ]
            [colors.append(self.getColorByValue(val)) for x, y, val in polygon]
        return polygons, normals, colors

    def getColorByValue(self, value):
        hue = 120 * value / 360
        color = QColor.fromHslF(hue, 1, 0.5)
        color_vec = QVector4D(color.redF(), color.greenF(), color.blueF(), 0.5)
        return color_vec

    def swapPoint(self, lon, lat, value):
        return lon, value, lat

    def swapPoints(self, polygon):
        return [(lon, value, lat) for lon, lat, value in polygon]

    def prepareNormalLines(self, polygons, normals, colors):  # DEBUG
        norm_i = 0
        start = len(self.coords_array)
        for polygon in polygons:
            point = polygon[0]
            normal = normals[norm_i]
            # color = colors[norm_i]
            color = QVector4D(1, 1, 1, 1)
            point_2 = QVector3D(*point) + normal * 0.04
            point_2 = (point_2.x(), point_2.y(), point_2.z())
            self.prepareLine((point, point_2), [color] * 2)
            norm_i += len(polygon)
        end = len(self.coords_array)
        return start, end

    def preparePolygons(self, polygons, normals, colors, start_index=None):
        assert len(normals) == len(colors)
        coords_array = []
        [[coords_array.append(list(p)) for p in polygon]
         for polygon in polygons]
        start = len(self.coords_array)
        if start_index is None:
            self.coords_array += coords_array
            self.normals += normals
            self.colors += colors
        else:
            for i, j in enumerate(
                    range(start_index, start_index + len(coords_array))):
                self.coords_array[j] = coords_array[i]
                self.normals[j] = normals[i]
                self.colors[j] = colors[i]
        end = len(self.coords_array)
        return start, end

    # LINES
    def getGrid(self):
        assert self.processor.min_lat != self.processor.max_lat \
            and self.processor.min_val != self.processor.max_val
        value = self.processor.min_val - \
            (self.processor.max_val - self.processor.min_val) * 0.1

        lines = []
        for lat in np.linspace(self.processor.min_lat, self.processor.max_lat,
                               self.grid_freq):
            line = ((self.processor.min_lon, lat, value),
                    (self.processor.max_lon, lat, value))
            lines.append(line)
        for lon in np.linspace(self.processor.min_lon, self.processor.max_lon,
                               self.grid_freq):
            line = ((lon, self.processor.min_lat, value),
                    (lon, self.processor.max_lat, value))
            lines.append(line)

        # lines.append(((self.min_lon, self.min_lat, self.min_val),
        #              (self.min_lon, self.min_lat, self.max_val)))

        lines = [
            self.swapPoints(self.processor.normalizePoints(line))
            for line in lines
        ]
        line_colors = [(self.grid_color, self.grid_color) for _ in lines]
        return lines, line_colors

    def getContour(self):
        lev_lines = self.processor.get_contour(levels=self.contour_levels)
        contour = []
        for level, line in lev_lines:
            line = [(self.processor.normalizeLon(lon),
                     self.processor.normalizeValue(level + 10),
                     self.processor.normalizeLat(lat)) for lon, lat in line]
            colors = [self.contour_color] * len(line)
            contour.append(self.prepareLine(line, colors))
        return contour

    def getLightLines(self):
        if self.processor.max_val == self.processor.min_val:
            v = 0
        else:
            v = self.processor.normalizeValue(
                self.processor.min_val -
                (self.processor.max_val - self.processor.min_val) * 0.1)
        lines = (((self.light_pos.x(), v, -100), (self.light_pos.x(), v, 100)),
                 ((-100, v, self.light_pos.z()), (100, v, self.light_pos.z())),
                 ((self.light_pos.x(), v, self.light_pos.z()),
                  (self.light_pos.x(), self.light_pos.y(),
                   self.light_pos.z())))
        line_colors = [(self.light_line_color, self.light_line_color)
                       for _ in lines]
        return lines, line_colors

    def prepareLines(self, lines, line_colors, start_index=None):
        assert len(lines) == len(line_colors)
        if start_index is None:
            data = [None] * len(lines)
        else:
            data = []
            sum_len = 0
            for line in lines:
                data.append(start_index + sum_len)
                sum_len += len(line)

        result = [
            self.prepareLine(line, colors, datum)
            for line, colors, datum in zip(lines, line_colors, data)
        ]
        return result[0][0], result[-1][1]

    def prepareLine(self, line, colors, start_index=None):
        assert len(line) == len(colors)
        if start_index is None:
            start = len(self.coords_array)
            self.coords_array += line
            self.colors += colors
            self.normals += [QVector3D(0, 1, 0)] * len(line)
            end = len(self.coords_array)
        else:
            start = start_index
            for i, j in enumerate(range(start_index, start_index + len(line))):
                self.coords_array[j] = line[i]
                self.colors[j] = colors[i]
            end = start_index + len(line)
        return start, end

    # ==================== SCENE PREPARATION ====================
    def initializeGL(self):
        GL.glClearColor(0.1, 0.1, 0.1, 1.0)
        GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
        self.setUpShaders()
        self.initVertexArrays()

    def setUpShaders(self):
        self.shaders.addShaderFromSourceFile(
            QOpenGLShader.Vertex, resource_path('shaders/shader.vert'))
        self.shaders.addShaderFromSourceFile(
            QOpenGLShader.Fragment, resource_path('shaders/shader.frag'))
        self.shaders.link()
        self.shaders.bind()

    def initVertexArrays(self):
        assert len(self.coords_array) == len(self.colors) == len(self.normals)
        GL.glVertexPointer(3, GL.GL_FLOAT, 0, self.coords_array)
        self.shaders.setAttributeArray("v_color", self.colors)
        self.shaders.enableAttributeArray("v_color")
        self.shaders.setAttributeArray("v_normal", self.normals)
        self.shaders.enableAttributeArray("v_normal")

    # ==================== UPDATING STUFF ====================
    def updateGL(func):
        def wrapper(self, *args, **kwargs):
            res = func(self, *args, **kwargs)
            self.openGLWidget.update()
            return res

        return wrapper

    def updateCameraInfo(func):
        def wrapper(self, *args, **kwargs):
            res = func(self, *args, **kwargs)
            self.elevationWidget.updatePos(
                self.processor.denormalizeValue(self.camera_pos.y()))
            self.minimapWidget.updateCameraInfo(self.camera_pos,
                                                self.camera_rot)
            return res

        return wrapper

    def updateLightData(self):
        polygons, normals, colors = self.getLightSourceCoords()
        self.preparePolygons(polygons, normals, colors, self.light_data[0])
        lines, colors = self.getLightLines()
        self.prepareLines(lines, colors, self.light_lines_data[0])
        GL.glVertexPointer(3, GL.GL_FLOAT, 0, self.coords_array)
        self.update_light = False

    def updateBuffer(self):
        self.prepareScene()
        self.initVertexArrays()
        self.update_buffer = False

    # ==================== ACTUAL DRAWING ====================
    def paintGL(self):
        self.loadScene()
        if self.update_light:
            self.updateLightData()
        if self.update_buffer:
            self.updateBuffer()
        self.updateMatrices()
        self.updateParams()
        self.drawScene()

    def loadScene(self):
        width, height = self.openGLWidget.width(), self.openGLWidget.height()
        view = max(width, height)
        GL.glViewport(int((width - view) / 2), int((height - view) / 2), view,
                      view)
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LEQUAL)

    def updateMatrices(self):
        proj = QMatrix4x4()
        if self.perspectiveRadioButton.isChecked():
            proj.frustum(-0.25, 0.25, -0.3, 0.2, 0.7, 20)
        else:
            proj.ortho(-0.25, 0.25, -0.1, 0.4, 0.7, 20)
        modelview = QMatrix4x4()
        modelview.lookAt(self.camera_pos, self.camera_pos + self.camera_rot,
                         QVector3D(0, 1, 0))
        self.shaders.setUniformValue("ModelViewMatrix", modelview)
        self.shaders.setUniformValue("MVP", proj * modelview)

    def updateParams(self):
        self.shaders.setUniformValue("LightPos", self.light_pos)
        self.shaders.setUniformValue("ambientStrength", self.ambient)
        self.shaders.setUniformValue("diffuseStrength", self.diffuse)
        self.shaders.setUniformValue("alpha", self.alpha)
        self.shaders.setUniformValue("center", self.center)
        self.shaders.setUniformValue("scale", self.scale_vec)

    def drawScene(self):
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        self.shaders.setUniformValue('scaleEnabled', False)
        self.shaders.setUniformValue('phongModel', False)
        self.drawPreparedPolygons(*self.light_data)
        if self.show_light_lines:
            self.drawPreparedLines(*self.light_lines_data)
        # self.shaders.setUniformValue('scaleEnabled', True)
        if self.show_grid:
            self.drawPreparedLines(*self.grid_data)
        # self.drawPreparedLines(*self.normal_data)
        self.shaders.setUniformValue('phongModel', True)
        self.drawPreparedPolygons(*self.map_data)
        if self.show_contour:
            self.shaders.setUniformValue('phongModel', False)
            self.drawPreparedLineStrips(self.contour_data)

    def drawPreparedPolygons(self, start, end):
        for i in range(start, end, 4):
            GL.glDrawArrays(GL.GL_POLYGON, i, 4)

    def drawPreparedLines(self, start, end):
        GL.glDrawArrays(GL.GL_LINES, start, end - start)

    def drawPreparedLineStrips(self, arr):
        for start, end in arr:
            GL.glDrawArrays(GL.GL_LINE_STRIP, start, end - start)

    # ==================== CONTROLS ====================
    def setupControls(self):
        # Camera
        self.moveCameraUp.clicked.connect(lambda: self.moveCamera(az=1))
        self.moveCameraDown.clicked.connect(lambda: self.moveCamera(az=-1))
        self.moveCameraLeft.clicked.connect(lambda: self.moveCamera(pol=1))
        self.moveCameraRight.clicked.connect(lambda: self.moveCamera(pol=-1))
        self.moveCameraForward.clicked.connect(lambda: self.moveCamera(z=-1))
        self.moveCameraBackward.clicked.connect(lambda: self.moveCamera(z=1))

        # Scaling
        self.xScaleSpinBox.valueChanged.connect(lambda x: self.scaleView(x=x))
        self.yScaleSpinBox.valueChanged.connect(lambda y: self.scaleView(y=y))
        self.zScaleSpinBox.valueChanged.connect(lambda z: self.scaleView(z=z))

        # Light
        self.ambientSlider.valueChanged.connect(lambda ambient: self.setLight(
            ambient=ambient / 100))
        self.diffuseSlider.valueChanged.connect(lambda diffuse: self.setLight(
            diffuse=diffuse / 100))

        self.xLightSpinBox.valueChanged.connect(lambda x: self.setLight(x=x))
        self.yLightSpinBox.valueChanged.connect(lambda y: self.setLight(y=y))
        self.zLightSpinBox.valueChanged.connect(lambda z: self.setLight(z=z))

        # Display
        self.gridCheckBox.toggled.connect(lambda g: self.setGrid(show=g))
        self.contourCheckBox.toggled.connect(lambda c: self.setContour(show=c))
        self.contourLevelSpinBox.valueChanged.connect(lambda l: self.
                                                      setContour(levels=l))

        # Misc
        self.alphaSlider.valueChanged.connect(lambda alpha: self.setDisplay(
            alpha / 100))
        self.actionGrabKeyboard.toggled.connect(self.toggleGrabKeyboard)
        self.actionGrabMouse.toggled.connect(self.toggleGrabMouse)
        self.orhogonalRadioButton.clicked.connect(self.openGLWidget.update)
        self.perspectiveRadioButton.clicked.connect(self.openGLWidget.update)

        self.openGLWidget.mousePressEvent \
            = lambda event: self.actionGrabMouse.setChecked(True)

    def keyPressed(self, event):
        key = event.key()
        camera_dict = {
            Qt.Key_W: {
                'z': 1
            },
            Qt.Key_S: {
                'z': -1
            },
            Qt.Key_A: {
                'x': -1
            },
            Qt.Key_D: {
                'x': 1
            },
            Qt.Key_Z: {
                'y': 1
            },
            Qt.Key_X: {
                'y': -1
            },
            Qt.Key_Up: {
                'az': 1
            },
            Qt.Key_Down: {
                'az': -1
            },
            Qt.Key_Left: {
                'pol': 1
            },
            Qt.Key_Right: {
                'pol': -1
            }
        }
        self.moveCamera(**camera_dict.get(key, {}))
        if key == Qt.Key_Escape:
            self.actionGrabMouse.setChecked(False)
            self.actionGrabKeyboard.setChecked(False)

    def mouseMoved(self, event):
        az_sensivity = 0.03
        pol_sensivity = 0.03
        if self.mouse_grabbed:
            delta = event.globalPos() - self.mouse_center
            QCursor.setPos(self.mouse_center)
            self.moveCamera(az=delta.y() * az_sensivity,
                            pol=delta.x() * pol_sensivity)
        else:
            super().mouseMoveEvent(event)

    @updateGL
    @updateCameraInfo
    def moveCamera(self, az=0, pol=0, x=0, y=0, z=0):
        move_coef = 0.1
        rot_coef = 2

        rot_matr = QMatrix4x4()
        rot_matr.rotate(-90, 0, 1, 0)
        rot_vec = QVector3D(self.camera_rot)
        rot_vec.setY(0)
        rot_vec = rot_matr * rot_vec

        if az != 0 or pol != 0:
            rot_matr = QMatrix4x4()
            rot_matr.rotate(rot_coef * az, rot_vec)
            rot_matr.rotate(rot_coef * pol, 0, 1, 0)
            self.camera_rot = rot_matr * self.camera_rot
        if z:
            self.camera_pos += move_coef * self.camera_rot * z
        if x:
            self.camera_pos += rot_vec * move_coef * x
        if y:
            self.camera_pos.setY(self.camera_pos.y() + y * move_coef)

    @updateGL
    def scaleView(self, x=None, y=None, z=None):
        if x:
            self.scale_vec.setX(x)
        if y:
            self.scale_vec.setY(y)
        if z:
            self.scale_vec.setZ(z)

    @updateGL
    def setLight(self, x=None, y=None, z=None, ambient=None, diffuse=None):
        if x:
            self.light_pos.setX(x)
        if y:
            self.light_pos.setY(y)
        if z:
            self.light_pos.setZ(z)
        if ambient:
            self.ambient = ambient
        if diffuse:
            self.diffuse = diffuse
        self.update_light = True

    @updateGL
    def setDisplay(self, alpha=None, invisible=None):
        if invisible:
            self.invisible = invisible
        if alpha:
            self.alpha = alpha

    @updateGL
    def setGrid(self, show):
        self.show_grid = show
        self.update_buffer = True

    @updateGL
    def setContour(self, show=None, levels=None):
        if show is not None:
            self.show_contour = show
            self.update_buffer = True
        if levels is not None:
            self.contour_levels = levels
            if self.show_contour:
                self.update_buffer = True

    def toggleGrabKeyboard(self, grab: bool):
        if grab:
            self.grabKeyboard()
        else:
            self.releaseKeyboard()

    def toggleGrabMouse(self, grab: bool):
        self.actionGrabKeyboard.setChecked(grab)
        self.mouse_grabbed = grab
        if grab:
            self.setCursor(Qt.BlankCursor)
            self.mouse_center = self.getMouseCenter()
            QCursor.setPos(self.mouse_center)
            self.setMouseTracking(True)
            self.grabMouse()
        else:
            self.setCursor(Qt.ArrowCursor)
            self.setMouseTracking(False)
            self.releaseMouse()

    def getMouseCenter(self):
        w, h = self.openGLWidget.width(), self.openGLWidget.height()
        local_center = QPoint(w / 2, h / 2)
        global_center = self.mapToGlobal(self.openGLWidget.pos()) \
            + local_center
        return global_center
