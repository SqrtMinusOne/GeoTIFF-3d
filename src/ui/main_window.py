import numpy as np
from OpenGL import GL
from PyQt5.QtCore import QPoint, Qt
from PyQt5.QtGui import (QColor, QCursor, QMatrix4x4, QOpenGLShader,
                         QOpenGLShaderProgram, QVector3D, QVector4D)
from PyQt5.QtWidgets import QMainWindow

from ui_compiled.mainwindow import Ui_MainWindow

__all__ = ['MainWindow']


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, processor, df, parent=None):
        super().__init__(parent)
        self.processor = processor
        # Data
        self.df = df
        self.min_val = self.max_val = 0
        self.min_lon = self.max_lon = self.min_lat = self.max_lat = 0
        self.val_lon_proportion = 1  # TODO?

        # UI
        self.setupUi(self)
        self.setupControls()
        self.keyPressEvent = self.keyPressed
        self.mouseMoveEvent = self.mouseMoved
        self.tabifyDockWidget(self.lightDockWidget, self.controlsDockWidget)
        self.lightDockWidget.raise_()

        # Control & Display
        self.mouse_grabbed = False

        self.camera_pos = QVector3D(0, 0, 4)
        self.camera_rot = QVector3D(0, 0, -1)
        self.scale_vec = QVector3D(1, 1, 1)

        self.light_pos = QVector3D(self.xLightSpinBox.value(),
                                   self.yLightSpinBox.value(),
                                   self.zLightSpinBox.value())
        self.ambient = self.ambientSlider.value() / 100
        self.diffuse = self.diffuseSlider.value() / 100
        self.alpha = self.alphaSlider.value() / 100
        self.draw_invisible = self.invisibleCheckBox.isChecked()

        # Drawing
        self.normals = []
        self.colors = []
        self.coords_array = []
        self.update_light = False

        self.grid_freq = 10

        self.grid_color = QVector4D(1, 1, 1, 1)
        self.contour_color = QVector4D(1, 1, 1, 1)

        self.prepareScene()

        self.shaders = QOpenGLShaderProgram()
        self.openGLWidget.initializeGL = self.initializeGL
        self.openGLWidget.paintGL = self.paintGL

    # ==================== PREPARATION ====================
    def prepareScene(self):
        polygons, normals, colors = self.getLightSourceCoords()
        self.light = self.preparePolygons(polygons, normals, colors)
        polygons, normals, colors = self.getMapPolygons()
        self.map_ = self.preparePolygons(polygons, normals, colors)
        lines, line_colors = self.getGrid()
        self.grid = self.prepareLines(lines, line_colors)
        self.contour = self.getContour()

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
        polygons = [[p * 0.1 + center for p in side] for side in polygons]

        colors = []
        [
            colors.append(QVector4D(1, 153 / 255, 0, 0.5) * self.diffuse)
            for _ in range(len(normals_vec))
        ]
        return polygons, normals_vec, colors

    def getMapPolygons(self):
        self.min_lon = min(self.df['x'])
        self.max_lon = max(self.df['x'])
        self.min_lat = min(self.df['y'])
        self.max_lat = max(self.df['y'])
        self.min_val = min(self.df['value'])
        self.max_val = max(self.df['value'])
        polygons, normals, colors = [], [], []
        for polygon, normal in self.processor.polygon_generator(self.df):
            polygons.append(self.swapPoints(self.normalizePoints(polygon)))
            [normals.append(QVector3D(*normal)) for _ in polygon]
            [colors.append(self.getColorByValue(val)) for x, y, val in polygon]
        return polygons, normals, colors

    def getColorByValue(self, value):
        value = (self.max_val - value) / (self.max_val - self.min_val)
        hue = 120 * value / 360
        color = QColor.fromHslF(hue, 1, 0.5)
        color_vec = QVector4D(color.redF(), color.greenF(), color.blueF(), 0.5)
        return color_vec

    def normalizePoints(self, polygon):
        normalized = []
        for lon, lat, value in polygon:
            lon_ = self.normalizeLon(lon)
            lat_ = self.normalizeLat(lat)
            value_ = self.normalizeValue(value) * self.val_lon_proportion
            normalized.append((lon_, lat_, value_))
        return normalized

    def swapPoints(self, polygon):
        return [(lon, value, lat) for lon, lat, value in polygon]

    def normalizeLat(self, lat):
        return (lat - self.min_lat) / (self.max_lat - self.min_lat)

    def normalizeLon(self, lon):
        return (lon - self.min_lon) / (self.max_lon - self.min_lon)

    def normalizeValue(self, value):
        return (value - self.min_val) / (self.max_val - self.min_val)

    def preparePolygons(self, polygons, normals, colors, start_index=None):
        coords_array = []
        [[coords_array.append(list(p)) for p in polygon]
         for polygon in polygons]
        start = len(self.coords_array)
        if start_index is None:
            self.coords_array += coords_array
            self.normals += normals
            self.colors += colors
        else:
            for i in range(start_index, start_index + len(coords_array)):
                self.coords_array[i] = coords_array[i]
                self.normals[i] = normals[i]
                self.colors[i] = colors[i]
        end = len(self.coords_array)
        return start, end

    # LINES
    def getGrid(self):
        assert self.min_lat != self.max_lat and self.min_val != self.max_val
        value = self.min_val - (self.max_val - self.min_val) * 0.1

        lines = []
        for lat in np.linspace(self.min_lat, self.max_lat, self.grid_freq):
            line = ((self.min_lon, lat, value), (self.max_lon, lat, value))
            lines.append(line)
        for lon in np.linspace(self.min_lon, self.max_lon, self.grid_freq):
            line = ((lon, self.min_lat, value), (lon, self.max_lat, value))
            lines.append(line)

        # lines.append(((self.min_lon, self.min_lat, self.min_val),
        #              (self.min_lon, self.min_lat, self.max_val)))

        lines = [self.swapPoints(self.normalizePoints(line)) for line in lines]
        line_colors = [(self.grid_color, self.grid_color)
                       for _ in lines]
        return lines, line_colors

    def getContour(self):
        lev_lines = self.processor.get_contour(levels=20)
        contour = []
        for level, line in lev_lines:
            line = [(self.normalizeLon(lon),
                     self.normalizeValue(level + 10),
                     self.normalizeLat(lat)) for lon, lat in line]
            colors = [self.contour_color] * len(line)
            contour.append(self.prepareLine(line, colors))
        return contour

    def prepareLines(self, lines, line_colors):
        assert len(lines) == len(line_colors)
        start = len(self.coords_array)
        [
            self.prepareLine(line, colors)
            for line, colors in zip(lines, line_colors)
        ]
        end = len(self.coords_array)
        return start, end

    def prepareLine(self, line, colors):
        assert len(line) == len(colors)
        start = len(self.coords_array)
        [self.coords_array.append(p) for p in line]
        [self.colors.append(c) for c in colors]
        [self.normals.append(QVector3D(0, 1, 0)) for _ in line]
        end = len(self.coords_array)
        return start, end

    # ==================== SCENE PREPARATION ====================
    def initializeGL(self):
        GL.glClearColor(0.1, 0.1, 0.1, 1.0)
        self.setUpShaders()
        self.initVertexArrays()

    def setUpShaders(self):
        self.shaders.addShaderFromSourceFile(QOpenGLShader.Vertex,
                                             'shaders/shader.vert')
        self.shaders.addShaderFromSourceFile(QOpenGLShader.Fragment,
                                             'shaders/shader.frag')
        self.shaders.link()
        self.shaders.bind()

        self.shaders.setAttributeArray("v_color", self.colors)
        self.shaders.enableAttributeArray("v_color")
        self.shaders.setAttributeArray("v_normal", self.normals)
        self.shaders.enableAttributeArray("v_normal")

    def initVertexArrays(self):
        GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
        GL.glVertexPointer(3, GL.GL_FLOAT, 0, self.coords_array)
        assert len(self.coords_array) == len(self.colors) == len(self.normals)

    # ==================== UPDATING STUFF ====================
    def updateGL(func):
        def wrapper(self, *args, **kwargs):
            res = func(self, *args, **kwargs)
            self.openGLWidget.update()
            return res

        return wrapper

    def updateLightData(self):
        polygons, normals, colors = self.getLightSourceCoords()
        self.preparePolygons(polygons, normals, colors, self.light[0])
        GL.glVertexPointer(3, GL.GL_FLOAT, 0, self.coords_array)
        self.update_light = False

    # ==================== ACTUAL DRAWING ====================
    def paintGL(self):
        self.loadScene()
        if self.update_light:
            self.updateLightData()
        self.updateMatrices()
        self.updateParams()
        self.drawScene()

    def loadScene(self):
        width, height = self.openGLWidget.width(), self.openGLWidget.height()
        view = max(width, height)
        GL.glViewport(0, 0, view, view)
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        if self.draw_invisible or True:  # TODO
            GL.glEnable(GL.GL_DEPTH_TEST)
            GL.glDepthFunc(GL.GL_LEQUAL)
        else:
            GL.glDisable(GL.GL_DEPTH_TEST)

    def updateMatrices(self):
        proj = QMatrix4x4()
        if self.perspectiveRadioButton.isChecked():
            proj.frustum(-0.3, 1, -0.3, 1, 2, 20)
        else:
            proj.ortho(-0.3, 1, -0.3, 1, 2, 20)
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

    def drawScene(self):
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        self.shaders.setUniformValue('phongModel', False)
        self.drawPreparedPolygons(*self.light)
        self.drawPreparedLines(*self.grid)
        self.shaders.setUniformValue('phongModel', True)
        self.drawPreparedPolygons(*self.map_)
        self.shaders.setUniformValue('phongModel', False)
        self.drawPreparedLineStrips(self.contour)

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

        # Misc
        self.alphaSlider.valueChanged.connect(lambda alpha: self.setDisplay(
            alpha / 100))
        self.invisibleCheckBox.stateChanged.connect(
            lambda invisible: self.setDisplay(invisible=invisible))
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
