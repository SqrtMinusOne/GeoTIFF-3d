import sys

import numpy as np
from PyQt5.QtCore import QRect, QRectF, Qt, QLine, QPointF, QPoint
from PyQt5.QtGui import QLinearGradient, QPainter, QPen, QPolygonF
from PyQt5.QtWidgets import (QApplication, QGraphicsItem, QGraphicsScene,
                             QGraphicsView)


__all__ = ['ElevationGraphWidget']


def getRect(widget):
    w, h = widget.width(), widget.height()
    y_ = 10
    h_ = h - 20
    w_ = int(h / 4)
    x_ = int((w - w_) / 2)
    return QRect(x_, y_, w_, h_)


class ElevationSquare(QGraphicsItem):
    def __init__(self, start, end, levels, parent=None):
        super().__init__(parent)
        self.start, self.end = start, end
        self.levels = levels
        self.rect = QRect(0, 0, 100, 100)

    def paint(self, painter: QPainter, option, widget=None):
        painter.setPen(QPen(Qt.black, 0))
        self.rect = getRect(widget)

        gradient = QLinearGradient(self.rect.topLeft(), self.rect.bottomLeft())
        gradient.setColorAt(0, Qt.red)
        gradient.setColorAt(1, Qt.green)
        painter.setBrush(gradient)
        painter.drawRect(self.rect)

        metrics = painter.fontMetrics()
        for level in self.levels:
            text = str(int(level))
            w, h = metrics.width(text), metrics.height()
            y = self.rect.height() - (level - self.start) / (
                self.end -
                self.start) * self.rect.height() + self.rect.y() - h / 2
            x = self.rect.x() - w - 10
            text_rect = QRectF(x, y, w, h)
            painter.drawText(text_rect, Qt.AlignRight, text)

    def boundingRect(self):
        adjust = 2
        return QRectF(self.rect.x() - adjust,
                      self.rect.y() - adjust,
                      self.rect.width() + adjust,
                      self.rect.height() + adjust)


class CameraTri(QGraphicsItem):
    def __init__(self, start, end, pos, parent=None):
        super().__init__(parent)
        self.start, self.end = start, end
        self.pos = pos
        self.line = QLine(0, 0, 100, 0)

    def updatePos(self, pos):
        self.pos = pos
        self.update()

    def getPoint(self):
        if self.pos < self.start:
            return self.line.p1()
        elif self.pos > self.end:
            return self.line.p2()
        else:
            c = (self.pos - self.start) / (self.end - self.start)
            return self.line.p1() * (1 - c) + self.line.p2() * c

    def paint(self, painter: QPainter, option, widget=None):
        rect = getRect(widget)
        delta = QPoint(5, 0)
        self.line = QLine(rect.bottomRight() + delta * 2,
                          rect.topRight() + delta * 2)
        point = self.getPoint()

        painter.setPen(QPen(Qt.black, 0))
        painter.setBrush(Qt.black)
        offset_h = QPointF(10, 0)
        offset_v = QPointF(0, 10)
        points = QPolygonF((
            QPointF(point),
            QPointF(point + offset_h - offset_v),
            QPointF(point + offset_h + offset_v),
            QPointF(point)
        ))
        painter.drawPolygon(points, 4)

    def boundingRect(self):
        offset = QPointF(20, 0)
        return QRectF(self.line.p1() - offset, self.line.p2() + offset)


class ElevationGraphWidget(QGraphicsView):
    def __init__(self, start, end, pos, width=240, height=240,
                 levels=None, parent=None):
        super().__init__(parent)
        self.start, self.end = start, end
        self.pos = pos
        self.levels = np.linspace(start, end, 5) if levels is None else levels
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

    def initContent(self):
        self.square = ElevationSquare(self.start, self.end, self.levels)
        self.tri = CameraTri(self.start, self.end, self.pos)
        self.scene().addItem(self.square)
        self.scene().addItem(self.tri)

    def updatePos(self, pos):
        self.tri.updatePos(int(pos))
        self.scene().update()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = ElevationGraphWidget(0, 1000, 500)
    widget.show()
    app.exec_()
