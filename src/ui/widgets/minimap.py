import numpy as np
from matplotlib import pyplot as plt
from PyQt5.QtCore import QPoint, QRect, QRectF, Qt
from PyQt5.QtGui import QPainter, QPen, QVector3D
from PyQt5.QtWidgets import QGraphicsItem, QGraphicsScene, QGraphicsView

from api import cart2pol, pol2cart

__all__ = ['MinimapGraphWidget']


def getRect(widget):
    w, h = widget.width(), widget.height()
    offset = 20
    return QRect(offset, offset, w - offset * 2, h - offset * 2)


class CameraItem(QGraphicsItem):
    """An item to show a camera and it's angle"""

    def __init__(self, processor, pos: QVector3D, rot: QVector3D, parent=None):
        super().__init__(parent)
        self.processor = processor
        self.pos = pos
        self.rot = QVector3D(rot)
        self.rot.setY(0)
        self.rot.normalize()
        self.rect = QRectF(0, 0, 10, 10)
        self.setZValue(11)

    def updateCameraInfo(self, pos=None, rot=None):
        if pos:
            self.pos = pos
        if rot:
            self.rot = QVector3D(rot)
            self.rot.setY(0)
            self.rot.normalize()

    def paint(self, painter: QPainter, option, widget=None):
        def normalize_arc_angle(angle):
            while angle > 16 * 180:
                angle -= 16 * 360
            while angle < -16 * 180:
                angle += 16 * 360
            return angle

        rect = getRect(widget)
        x = self.pos.x() * rect.width() + rect.x()
        y = self.pos.z() * rect.height() + rect.y()
        x, y = max(x, 0), max(y, 0)
        x, y = min(x, widget.width()), min(y, widget.height())

        circ_rad = 3
        arr_rad = 20
        dangle = np.pi / 180 * 30

        point = QPoint(x, y)
        angle = cart2pol(self.rot.x(), self.rot.z())[1]
        delta_1 = QPoint(*pol2cart(arr_rad, angle - dangle))
        delta_2 = QPoint(*pol2cart(arr_rad, angle + dangle))

        arc_rect = QRect(point - QPoint(arr_rad, arr_rad),
                         point + QPoint(arr_rad, arr_rad))
        self.rect = arc_rect
        arc_start = (-angle + dangle) / np.pi * 180 * 16
        arc_start = normalize_arc_angle(arc_start)
        arc_span = -60 * 16

        painter.setPen(QPen(Qt.black, 0))
        painter.setBrush(Qt.black)
        painter.drawEllipse(point, circ_rad, circ_rad)

        painter.setBrush(Qt.NoBrush)
        # painter.drawEllipse(point, arr_rad, arr_rad)
        painter.drawLine(point, point + delta_1)
        painter.drawLine(point, point + delta_2)
        # painter.drawRect(arc_rect)
        painter.drawArc(arc_rect, arc_start, arc_span)

    def boundingRect(self):
        return QRectF(self.rect)


class MinimapGraphWidget(QGraphicsView):
    """The widget to show a minimap with a camera position"""

    def __init__(self,
                 processor,
                 position: QVector3D,
                 rotation: QVector3D,
                 width=240,
                 height=240,
                 parent=None):
        super().__init__(parent)
        self.processor = processor
        self.pos = position
        self.rot = rotation
        self.resize(width, height)

        scene = QGraphicsScene(self)
        scene.setItemIndexMethod(QGraphicsScene.NoIndex)
        scene.setSceneRect(0, 0, self.width(), self.height())
        self.setScene(scene)
        self.setCacheMode(self.CacheBackground)
        self.setViewportUpdateMode(self.BoundingRectViewportUpdate)
        self.setRenderHint(QPainter.Antialiasing)
        self.setTransformationAnchor(self.AnchorUnderMouse)
        self.setSizeAdjustPolicy(self.AdjustToContents)
        self.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setMinimumSize(60, 60)

        self.initContent()
        self.resizeEvent = self.onResize
        self.cnv_x_thresholds = {
            0: {
                'left': 0.01,
                'right': 0.99
            },
            200: {
                'left': 0.2,
                'right': 0.8
            },
            520: {
                'left': 0.1,
                'right': 0.9
            }
        }
        self.cnv_y_thresholds = {
            0: {
                'bottom': 0.01,
                'top': 0.99
            },
            200: {
                'bottom': 0.2,
                'top': 0.8
            },
            520: {
                'bottom': 0.1,
                'top': 0.9
            }
        }
        self.contour_kwargs = {'levels': 10, 'linewidths': 1}
        self.x_threshold = 0
        self.y_threshold = 0
        self.big_canvas = True

    def updateCameraInfo(self, pos, rot):
        self.camera.updateCameraInfo(pos, rot)
        self.scene().update()

    def getThresholds(self):
        rect = getRect(self)
        x_threshold = max([width for width in self.cnv_x_thresholds.keys()
                           if width <= rect.width()])
        y_threshold = max([height for height in self.cnv_y_thresholds.keys()
                           if height <= rect.height()])
        return x_threshold, y_threshold

    def onResize(self, event):
        rect = getRect(self)
        self.canvas.setGeometry(rect)
        x_threshold, y_threshold = self.getThresholds()
        if self.x_threshold != x_threshold or self.y_threshold != y_threshold:
            self.processor.ax.cla()
            self.processor.get_contour(plot=True, **self.contour_kwargs)
            if x_threshold == 0 or y_threshold == 0:
                plt.subplots_adjust(**self.cnv_x_thresholds[0],
                                    **self.cnv_y_thresholds[0])
                self.processor.ax.set_xticks([], [])
                self.processor.ax.set_yticks([], [])
            else:
                plt.subplots_adjust(
                    **self.cnv_x_thresholds[x_threshold],
                    **self.cnv_y_thresholds[y_threshold])
            self.x_threshold, self.y_threshold = x_threshold, y_threshold
        super().resizeEvent(event)

    def initContent(self):
        # self.map_ = MapRect(self.processor)
        self.camera = CameraItem(self.processor, self.pos, self.rot)

        self.canvas = self.processor.init_canvas()
        plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
        self.processor.get_contour(plot=True)
        self.processor.ax.set_xticks([], [])
        self.processor.ax.set_yticks([], [])
        rect = getRect(self)
        self.canvas.setGeometry(rect)

        # self.scene().addItem(self.map_)
        self.scene().addItem(self.camera)
        self.scene().addWidget(self.canvas)
