import numpy as np

from PyQt5.QtCore import Qt, QRect, QRectF, QPoint
from PyQt5.QtGui import QPainter, QPen, QVector3D
from PyQt5.QtWidgets import (QGraphicsItem, QGraphicsScene,
                             QGraphicsView)
from matplotlib import pyplot as plt

from api import cart2pol, pol2cart


__all__ = ['MinimapGraphWidget']


def getRect(widget):
    w, h = widget.width(), widget.height()
    offset = 20
    return QRect(offset, offset, w - offset * 2, h - offset * 2)


class MapRect(QGraphicsItem):
    def __init__(self, processor, parent=None):
        super().__init__(parent)
        self.processor = processor
        self.rect = QRect(10, 10, 90, 90)
        self.setZValue(10)

    def paint(self, painter: QPainter, option, widget=None):
        self.rect = getRect(widget)
        painter.setPen(QPen(Qt.black, 0))
        painter.drawRect(self.rect)

    def boundingRect(self):
        return QRectF(self.rect)


class CameraItem(QGraphicsItem):
    def __init__(self, processor, pos: QVector3D, rot: QVector3D,
                 parent=None):
        super().__init__(parent)
        self.processor = processor
        self.min_lon, self.max_lon, self.min_lat, self.max_lat \
            = processor.get_borders()
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
            while angle < - 16 * 180:
                angle += 16 * 360
            return angle

        rect = getRect(widget)
        x = self.pos.x() * rect.width() + rect.x()
        y = (1 - self.pos.z()) * rect.height() + rect.y()
        x, y = max(x, 0), max(y, 0)
        x, y = min(x, widget.width()), min(y, widget.height())

        circ_rad = 3
        arr_rad = 20
        dangle = np.pi / 180 * 30

        point = QPoint(x, y)
        angle = -cart2pol(self.rot.x(), self.rot.z())[1]
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
    def __init__(self, processor, position: QVector3D, rotation: QVector3D,
                 width=240, height=240, parent=None):
        super().__init__(parent)
        self.processor = processor
        self.min_lon, self.max_lon, self.min_lat, self.max_lat \
            = processor.get_borders()
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

    def updateCameraInfo(self, pos, rot):
        self.camera.updateCameraInfo(pos, rot)
        self.scene().update()

    def onResize(self, event):
        rect = getRect(self)
        self.canvas.setGeometry(rect)
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
