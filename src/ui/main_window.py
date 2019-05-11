from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtGui import QVector3D, QOpenGLShaderProgram, QOpenGLShader, \
    QMatrix4x4, QVector4D, QCursor
from OpenGL import GL
import numpy as np

from ui_compiled.mainwindow import Ui_MainWindow


__all__ = ['MainWindow']


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.setupControls()
        self.keyPressEvent = self.keyPressed
        self.mouseMoveEvent = self.mouseMoved
        self.tabifyDockWidget(self.lightDockWidget, self.controlsDockWidget)
        self.lightDockWidget.raise_()

        self.mouse_grabbed = False

        self.camera_pos = QVector3D(0, 0, 4)
        self.camera_rot = QVector3D(0, 0, -1)
        self.scale_vec = QVector3D(1, 1, 1)

        self.light_pos = QVector3D(
            self.xLightSpinBox.value(),
            self.yLightSpinBox.value(),
            self.zLightSpinBox.value()
        )
        self.ambient = self.ambientSlider.value() / 100
        self.diffuse = self.diffuseSlider.value() / 100
        self.alpha = self.alphaSlider.value() / 100
        self.draw_invisible = self.invisibleCheckBox.isChecked()

        self.shaders = QOpenGLShaderProgram()
        self.openGLWidget.initializeGL = self.initializeGL
        self.openGLWidget.paintGL = self.paintGL

    def updateGL(func):
        def wrapper(self, *args, **kwargs):
            res = func(self, *args, **kwargs)
            self.updateScene()
            return res
        return wrapper

    def updateScene(self):
        self.openGLWidget.update()

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
        self.ambientSlider.valueChanged.connect(
            lambda ambient: self.setLight(ambient=ambient / 100))
        self.diffuseSlider.valueChanged.connect(
            lambda diffuse: self.setLight(diffuse=diffuse / 100))

        self.xLightSpinBox.valueChanged.connect(lambda x: self.setLight(x=x))
        self.yLightSpinBox.valueChanged.connect(lambda y: self.setLight(y=y))
        self.zLightSpinBox.valueChanged.connect(lambda z: self.setLight(z=z))

        # Misc
        self.alphaSlider.valueChanged.connect(
            lambda alpha: self.setDisplay(alpha / 100))
        self.invisibleCheckBox.stateChanged.connect(
            lambda invisible: self.setDisplay(invisible=invisible))
        self.actionGrabKeyboard.toggled.connect(self.toggleGrabKeyboard)
        self.actionGrabMouse.toggled.connect(self.toggleGrabMouse)
        self.orhogonalRadioButton.clicked.connect(self.updateScene)
        self.perspectiveRadioButton.clicked.connect(self.updateScene)

    def keyPressed(self, event):
        key = event.key()
        camera_dict = {
            Qt.Key_W: {'z': 1},
            Qt.Key_S: {'z': -1},
            Qt.Key_A: {'x': -1},
            Qt.Key_D: {'x': 1},
            Qt.Key_Shift: {'y': 1},
            Qt.Key_Control: {'y': -1},
            Qt.Key_Up: {'az': 1},
            Qt.Key_Down: {'az': -1},
            Qt.Key_Left: {'pol': 1},
            Qt.Key_Right: {'pol': -1}
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
            self.moveCamera(az=delta.y()*az_sensivity,
                            pol=delta.x()*pol_sensivity)
        else:
            super().mouseMoveEvent(event)

    def initializeGL(self):
        GL.glClearColor(0.1, 0.1, 0.1, 1.0)
        self.setUpShaders()

    def setUpShaders(self):
        self.shaders.addShaderFromSourceFile(QOpenGLShader.Vertex,
                                             'shaders/shader.vert')
        self.shaders.addShaderFromSourceFile(QOpenGLShader.Fragment,
                                             'shaders/shader.frag')
        self.shaders.link()
        self.shaders.bind()
        self.updateMatrices()

    def paintGL(self):
        self.loadScene()
        self.updateMatrices()
        self.drawScene()

    def loadScene(self):
        width, height = self.openGLWidget.width(), self.openGLWidget.height()
        GL.glViewport(0, 0, width, height)
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        if self.draw_invisible:
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
        modelview.lookAt(
            self.camera_pos,
            self.camera_pos + self.camera_rot,
            QVector3D(0, 1, 0)
        )

        self.shaders.setUniformValue("ModelViewMatrix", modelview)
        self.shaders.setUniformValue("MVP", proj * modelview)
        self.shaders.setUniformValue("LightPos", self.light_pos)

    def drawScene(self):
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        self.drawLightSource()

    def drawLightSource(self):
        polygons, normals, colors = self.getLightSourceCoords()
        self.shaders.setUniformValue('phongModel', False)
        self.drawPolygons(polygons, normals, colors)

    def getLightSourceCoords(self):
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

        normals_vec = []
        [[normals_vec.append(QVector3D(*normals[i]))
          for _ in range(len(polygons[i]))]
         for i in range(len(polygons))]

        center = np.array((self.light_pos.x(),
                           self.light_pos.y(),
                           self.light_pos.z()))
        polygons = [[p * 0.1 + center for p in side] for side in polygons]

        colors = []
        [colors.append(QVector4D(1, 153 / 255, 0, 0.5) * self.diffuse)
         for _ in range(len(polygons) * 4)]
        return polygons, normals_vec, colors

    def drawPolygons(self, polygons, normals, colors):
        self.shaders.setAttributeArray("v_color", colors)
        self.shaders.enableAttributeArray("v_color")
        GL.glEnableClientState(GL.GL_VERTEX_ARRAY)

        coords_array = []
        [[coords_array.append(list(p)) for p in polygon]
         for polygon in polygons]
        len_array = [len(polygon) for polygon in polygons]
        GL.glVertexPointer(3, GL.GL_FLOAT, 0, coords_array)
        for i in range(len(polygons)):
            start_index = sum(len_array[:i])
            GL.glDrawArrays(GL.GL_POLYGON, start_index, len_array[i])

        GL.glDisableClientState(GL.GL_VERTEX_ARRAY)

    @updateGL
    def moveCamera(self, az=0, pol=0, x=0, y=0, z=0):
        move_coef = 0.1
        rot_coef = 2

        if az != 0 or pol != 0:
            rot_matr = QMatrix4x4()
            rot_matr.rotate(rot_coef * az, 1, 0, 0)
            rot_matr.rotate(rot_coef * pol, 0, 1, 0)
            self.camera_rot = rot_matr * self.camera_rot
        if z:
            self.camera_pos += move_coef * self.camera_rot * z
        if x:
            rot_matr = QMatrix4x4()
            rot_matr.rotate(-90, 0, 1, 0)
            self.camera_pos += rot_matr * self.camera_rot * move_coef * x
        if y:
            self.camera_pos.setY(self.camera_pos.y() + y * move_coef)

    def scaleView(self, x=None, y=None, z=None):
        if x:
            self.scale_vec.setX(x)
        if y:
            self.scale_vec.setY(y)
        if z:
            self.scale_vec.setZ(z)

    @updateGL
    def setLight(self, x=None, y=None, z=None, ambient=None,
                 diffuse=None):
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
