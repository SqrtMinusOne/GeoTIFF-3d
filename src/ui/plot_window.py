import threading
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtGui import QOpenGLShaderProgram, QOpenGLShader, QMatrix4x4, \
    QVector3D, QVector4D
from OpenGL import GL
from matplotlib import tri

from ui_compiled.main import Ui_MainWindow


__all__ = ['PlotWindow']


def cart2pol(x, y):
    r = np.sqrt(x ** 2 + y ** 2)
    ang = np.arctan2(y, x)
    return r, ang


def pol2cart(r, ang):
    x = r * np.cos(ang)
    y = r * np.sin(ang)
    return x, y


class PlotWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.screw_color = (223 / 255, 37 / 255, 0 / 255, 0.8)
        self.setupUi(self)
        self.getLightPos()
        self.mapControls()
        self.openGLWidget.initializeGL = self.initializeGL
        self.openGLWidget.paintGL = self.paintGL
        self.keyPressEvent = self.onKeyPressed
        self.objectAngleX, self.objectAngleY, self.objectAngleZ = 0, 0, 0
        self.object_center = QVector3D(0.5, 0.5, 0.5)
        self.camera_pos = QVector3D(0, 0, 4)
        self.scale_vec = QVector3D(1, 1, 1)
        self.ambient = 0.2
        self.diffuse = 0.8

        self.precision = 20
        self.draw_lines = False
        self.invisible = True

        self.mutex = threading.Lock()
        self.shaders = QOpenGLShaderProgram()

    def toggleGrabKeyboard(self, grab):
        if grab:
            self.grabKeyboard()
        else:
            self.releaseKeyboard()

    def getLightPos(self):
        self.light_pos = QVector3D(
            self.xLightSpinBox.value(),
            self.yLightSpinBox.value(),
            self.zLightSpinBox.value()
        )

    def loadScene(self):
        width, height = self.openGLWidget.width(), self.openGLWidget.height()
        GL.glViewport(0, 0, width, height)
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        if self.invisible:
            GL.glEnable(GL.GL_DEPTH_TEST)
            GL.glDepthFunc(GL.GL_LEQUAL)
        else:
            GL.glDisable(GL.GL_DEPTH_TEST)
        self.getLightPos()

    def initializeGL(self):
        GL.glClearColor(0.1, 0.1, 0.1, 1.0)
        self.setUpShaders()

    def paintGL(self):
        self.loadScene()
        with self.mutex:
            self.updateMatrices()
            self.drawStuff()

    def setUpShaders(self):
        self.shaders.addShaderFromSourceFile(QOpenGLShader.Vertex,
                                             'shaders/shader.vert')
        self.shaders.addShaderFromSourceFile(QOpenGLShader.Fragment,
                                             'shaders/shader.frag')
        self.shaders.link()
        self.shaders.bind()

        self.updateMatrices()

    def updateMatrices(self):
        proj = QMatrix4x4()
        if self.projectionComboBox.currentIndex() == 0:
            proj.frustum(-0.3, 1, -0.3, 1, 2, 20)
        else:
            proj.ortho(-0.3, 1, -0.3, 1, 2, 20)
        modelview = QMatrix4x4()
        modelview.lookAt(
            self.camera_pos,
            QVector3D(0, 0, 0),
            QVector3D(0, 1, 0)
        )

        self.shaders.setUniformValue("ModelViewMatrix", modelview)
        self.shaders.setUniformValue("MVP", proj * modelview)
        self.shaders.setUniformValue("LightPos", self.light_pos)

    def drawStuff(self):
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        # self.star_coords.clear()
        # [self.putStar(self.star_color, *param) for param in self.star_params]
        # self.drawFlag(self.flag_coords)
        # self.drawStars()
        self.drawCoordSystem()
        self.drawObject()
        self.drawLightSource()

    def getCubeCoords(self):
        polygons = np.array([
            ((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)),
            ((0, 0, 1), (0, 1, 1), (1, 1, 1), (1, 0, 1)),
            ((0, 0, 0), (0, 0, 1), (1, 0, 1), (1, 0, 0)),
            ((0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)),
            ((0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)),
            ((1, 0, 0), (1, 0, 1), (1, 1, 1), (1, 1, 0))
        ])
        normals = [
            (0, 0, -1),
            (0, 0, 1),
            (0, -1, 0),
            (0, 1, 0),
            (-1, 0, 0),
            (1, 0, 0)
        ]
        new_normals = []

        [[new_normals.append(QVector3D(*normals[i]))
          for _ in range(len(polygons[i]))] for i in range(len(polygons))]
        center = np.array(
            (self.light_pos.x(),
             self.light_pos.y(),
             self.light_pos.z()))
        polygons = [[p * 0.1 + center for p in side] for side in polygons]
        return polygons, new_normals, center

    def getCircleApprox(self, r, y, h, precision,  # REMOVE
                        map_proc=None, map_norm=None):
        map_proc = map_proc if map_proc is not None else lambda v: v
        map_norm = map_norm if map_norm is not None else map_proc
        polygons = []
        normals = []
        for i1, i2 in zip(range(precision+1), range(1, precision+1)):
            angle_1 = 2 * np.pi / precision * i1
            angle_2 = 2 * np.pi / precision * i2
            x1, z1 = pol2cart(r, angle_1)
            x2, z2 = pol2cart(r, angle_2)
            p1 = map_proc(QVector3D(x1, y, z1))
            p2 = map_proc(QVector3D(x1, y + h, z1))
            p3 = map_proc(QVector3D(x2, y + h, z2))
            p4 = map_proc(QVector3D(x2, y, z2))
            polygons.append([(p.x(), p.y(), p.z()) for p in (p1, p2, p3, p4)])

            xn, zn = pol2cart(1, (angle_1 + angle_2) / 2)
            normal = map_norm(QVector3D(xn, 0, zn))
            [normals.append((normal.x(), normal.y(), normal.z()))
             for _ in range(4)]
        return polygons, normals

    def getFlatCircleApprox(self, r1, r2, y, y_dir, p1, p2,  # REMOVE
                            map_proc=None, map_norm=None):
        map_proc = map_proc if map_proc is not None else lambda v: v
        map_norm = map_norm if map_norm is not None else map_proc
        polygons = []
        normals = []
        ang_space_1 = np.linspace(0, 2 * np.pi, p1+1)
        ang_space_2 = np.linspace(0, 2 * np.pi, p2+1)
        space = np.array([pol2cart(r1, a) for a in ang_space_1]
                         + [pol2cart(r2, a) for a in ang_space_2])
        x_s, z_s = space[:, 0], space[:, 1]
        tr = tri.Triangulation(x_s, z_s)
        x_tr, y_tr = x_s[tr.triangles].mean(
            axis=1), z_s[tr.triangles].mean(axis=1)

        def mask_func(x, y): return (
            lambda pol=cart2pol(x, y): (
                min(r1, r2) > pol[0]
            )
        )()
        mask_array = [mask_func(*point) for point in zip(x_tr, y_tr)]
        tr.set_mask(mask_array)
        normal = map_norm(QVector3D(0, y_dir, 0))
        for triangle in tr.get_masked_triangles():
            polygon = [map_proc(QVector3D(
                x_s[p], y, z_s[p]
            )) for p in triangle]
            polygons.append([(p.x(), p.y(), p.z()) for p in polygon])
            [normals.append((normal.x(), normal.y(), normal.z()))
             for _ in range(len(polygon))]
        return polygons, normals

    def map_point(self, point: QVector3D):
        matr = QMatrix4x4()
        matr.rotate(self.objectAngleX, QVector3D(1, 0, 0))
        matr.rotate(self.objectAngleY, QVector3D(0, 1, 0))
        matr.rotate(self.objectAngleZ, QVector3D(0, 0, 1))
        matr.scale(self.scale_vec)
#        matr.translate(self.object_center)

        point = point * matr

        return point + self.object_center

    def map_normal(self, normal: QVector3D):
        matr = QMatrix4x4()
        matr.rotate(self.objectAngleX, QVector3D(1, 0, 0))
        matr.rotate(self.objectAngleY, QVector3D(0, 1, 0))
        matr.rotate(self.objectAngleZ, QVector3D(0, 0, 1))

        normal = normal * matr

        return normal

    def getObjectCoords(self, precision=20):  # FIX
        def get_carving(norm_r, carve_r, y_start, carve_h, num, precision):
            vert, hor = [], []
            for i in range(0, (num + 1)*2, 2):
                vert.append(
                    (norm_r, y_start + i * carve_h, carve_h, precision)
                )
            for i in range(1, num*2, 2):
                vert.append(
                    (carve_r, y_start + i * carve_h, carve_h, precision)
                )
            for i in range(1, (num+1)*2-1):
                hor.append(
                    (norm_r, carve_r, y_start + i * carve_h, -(i % 2) * 2 + 1,
                     precision, precision)
                )
            return vert, hor

        # center = QVector3D(0.5, 0.5, 0.5)
        map_proc = self.map_point
        map_norm = self.map_normal
        vert, hor = [], []
        vert, hor = get_carving(0.8, 0.7, 0, 0.05, 5, precision)
        v, h = get_carving(0.6, 0.5, 0.05*11, 0.05, 5, precision)
        vert += v
        hor += h
        vert += [
            (1, 0, 0.05*12, 6),
            (0.7, 0.05*12, 0.05*10, 6)
        ]
        hor += [
            (1, 0.8, 0, -1, 6, precision),
            (1, 0.7, 0.05*11, -1, 6, 6),
            (1, 0.7, 0.05*12, 1, 6, 6),
            (0.7, 0.5, 0.05*11, -1, 6, precision),
            (0.7, 0.6, 0.05*(11+11), 1, 6, precision)
        ]
        coords, normals = [], []
        for vertical in vert:
            c, n = self.getCircleApprox(*vertical, map_proc, map_norm)
            coords += c
            normals += n

        for horizontal in hor:
            c, n = self.getFlatCircleApprox(*horizontal, map_proc, map_norm)
            coords += c
            normals += n

        return coords, normals

    def drawObject(self):
        coords, normals = self.getObjectCoords(self.precision)
        screw_colors, line_colors = [], []
        [screw_colors.append(QVector4D(*self.screw_color))
         for _ in range(len(normals))]
        [line_colors.append(QVector4D(1, 1, 1, 1))
         for _ in range(len(normals))]

        self.shaders.setAttributeArray("v_color", screw_colors)
        self.shaders.enableAttributeArray("v_color")
        self.shaders.setAttributeArray("v_normal", normals)
        self.shaders.enableAttributeArray("v_normal")

        self.shaders.setUniformValue("ambientStrength", self.ambient)
        self.shaders.setUniformValue("diffuseStrength", self.diffuse)
        self.shaders.setUniformValue("phongModel", True)

        GL.glEnableClientState(GL.GL_VERTEX_ARRAY)

        coords_array = []
        [[coords_array.append(p) for p in polygon] for polygon in coords]
        len_array = [len(polygon) for polygon in coords]
        GL.glVertexPointer(3, GL.GL_FLOAT, 0, coords_array)
        for i in range(len(coords)):
            start_index = sum(len_array[:i])
            GL.glDrawArrays(GL.GL_POLYGON, start_index, len_array[i])

        if self.draw_lines:
            self.shaders.setAttributeArray("v_color", line_colors)
            for i in range(len(coords)):
                start_index = sum(len_array[:i])
                GL.glDrawArrays(GL.GL_LINE_LOOP, start_index, len_array[i])
        GL.glDisableClientState(GL.GL_VERTEX_ARRAY)

    def drawLightSource(self):
        coords, normals, center = self.getCubeCoords()
        source_colors = []
        [source_colors.append(QVector4D(1, 153 / 255, 0, 1) * self.diffuse)
         for _ in range(len(normals))]

        self.shaders.setAttributeArray("v_color", source_colors)
        self.shaders.enableAttributeArray("v_color")

        self.shaders.setUniformValue("phongModel", False)

        GL.glEnableClientState(GL.GL_VERTEX_ARRAY)

        coords_array = []
        [[coords_array.append(list(p)) for p in polygon] for polygon in coords]
        len_array = [len(polygon) for polygon in coords]
        GL.glVertexPointer(3, GL.GL_FLOAT, 0, coords_array)
        for i in range(len(coords)):
            start_index = sum(len_array[:i])
            GL.glDrawArrays(GL.GL_POLYGON, start_index, len_array[i])

        GL.glDisableClientState(GL.GL_VERTEX_ARRAY)

    def drawCoordSystem(self):
        coords = (
            ((0, 0, -100), (0, 0, 100)),
            ((-100, 0, 0), (100, 0, 0)),
            ((0, -100, 0), (0, 100, 0))
        )
        coords_array = []
        line_colors = []
        [line_colors.append(QVector4D(1, 1, 1, 1))
         for _ in range(len(coords) * 2)]
        [[coords_array.append(p) for p in line] for line in coords]

        GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
        self.shaders.setAttributeArray("v_color", line_colors)
        self.shaders.setUniformValue("phongModel", False)
        GL.glVertexPointer(3, GL.GL_FLOAT, 0, coords_array)
        for i in range(len(coords)):
            start_index = i * 2
            GL.glDrawArrays(GL.GL_LINES, start_index, 2)

        GL.glDisableClientState(GL.GL_VERTEX_ARRAY)

    def onKeyPressed(self, event):
        key = event.key()
        if key == Qt.Key_A:
            self.moveCamera(x=-1)
        elif key == Qt.Key_D:
            self.moveCamera(x=1)
        elif key == Qt.Key_W:
            self.moveCamera(z=1)
        elif key == Qt.Key_S:
            self.moveCamera(z=-1)
        elif key == Qt.Key_Z:
            self.moveCamera(y=1)
        elif key == Qt.Key_X:
            self.moveCamera(y=-1)
        elif key == Qt.Key_8:
            self.rotateObject(1, 0)
        elif key == Qt.Key_2:
            self.rotateObject(-1, 0)
        elif key == Qt.Key_4:
            self.rotateObject(0, -1)
        elif key == Qt.Key_6:
            self.rotateObject(0, 1)
        self.openGLWidget.update()

    def mapControls(self):
        # self.moveCameraUp.clicked.connect(lambda: self.moveCamera(y=1))
        # self.moveCameraDown.clicked.connect(lambda: self.moveCamera(y=-1))
        # self.moveCameraLeft.clicked.connect(lambda: self.moveCamera(x=-1))
        # self.moveCameraRight.clicked.connect(lambda: self.moveCamera(x=1))
        # self.moveCameraForward.clicked.connect(lambda: self.moveCamera(z=-1))
        # self.moveCameraBackward.clicked.connect(lambda: self.moveCamera(z=1))

        # self.rotateObjectUp.clicked.connect(lambda: self.rotateObject(x=-1))
        # self.rotateObjectDown.clicked.connect(lambda: self.rotateObject(x=1))
        # self.rotateObjectLeft.clicked.connect(
        #   lambda: self.rotateObject(y=-1))
        # self.rotateObjectRight.clicked.connect(
        #   lambda: self.rotateObject(y=1))
        # self.rotateObjectForward.clicked.connect(
        #     lambda: self.rotateObject(z=1))
        # self.rotateObjectBackward.clicked.connect(
        #     lambda: self.rotateObject(z=-1))

        # self.xScaleSpinBox.valueChanged.connect(lambda x:self.scaleView(x=x))
        # self.yScaleSpinBox.valueChanged.connect(lambda y:self.scaleView(y=y))
        # self.zScaleSpinBox.valueChanged.connect(lambda z:self.scaleView(z=z))

        # self.xCenterSpinBox.valueChanged.connect(
        #     lambda x: self.object_center.setX(x))
        # self.yCenterSpinBox.valueChanged.connect(
        #     lambda y: self.object_center.setY(y))
        # self.zCenterSpinBox.valueChanged.connect(
        #     lambda z: self.object_center.setZ(z))

        # self.precisionSlider.valueChanged.connect(
        #    lambda p: setattr(self, 'precision', p))
        self.grabKeyboardCheckBox.stateChanged.connect(
            self.toggleGrabKeyboard)
        self.ambientSlider.valueChanged.connect(
            lambda v: setattr(self, 'ambient', v / 100))
        self.diffuseSlider.valueChanged.connect(
            lambda v: setattr(self, 'diffuse', v / 100))
        self.alphaSlider.valueChanged.connect(
            lambda a: setattr(
                self,
                'screw_color',
                (self.screw_color[0],
                 self.screw_color[1],
                 self.screw_color[2],
                 a / 100)))

        # self.drawLinesCheckBox.stateChanged.connect(
        #    lambda s: setattr(self, 'draw_lines', s))
        self.invisibleCheckBox.stateChanged.connect(
            lambda s: setattr(self, 'invisible', s))

    def rotateObject(self, x=0, y=0, z=0):
        angle_move = 5
        self.objectAngleX = (self.objectAngleX + angle_move * x) % 360
        self.objectAngleY = (self.objectAngleY + angle_move * y) % 360
        self.objectAngleZ = (self.objectAngleZ + angle_move * z) % 360

    def moveCamera(self, x=0, y=0, z=0):
        camera_move = 0.2
        self.camera_pos.setX(self.camera_pos.x() + camera_move * x)
        self.camera_pos.setY(self.camera_pos.y() + camera_move * y)
        self.camera_pos.setZ(self.camera_pos.z() + camera_move * z)

    def scaleView(self, x=None, y=None, z=None):
        if x:
            self.scale_vec.setX(x)
        if y:
            self.scale_vec.setY(y)
        if z:
            self.scale_vec.setZ(z)
