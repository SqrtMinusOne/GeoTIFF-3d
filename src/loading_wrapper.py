from PyQt5.QtCore import pyqtSignal, QThread


__all__ = ['LoadingThread']


class LoadingThread(QThread):
    """This is base class of thread for using with LoadingWrapper
    The idea is to move some heavy operations to a special thread and show
    progress on the LoadingDialog.
    This actually decreases perfomance a bit because of GIL, but improves user
    experience"""
    updateStatus = pyqtSignal(str)  # Update status string
    updatePercent = pyqtSignal(int)  # Update a percent
    updateMaxPercent = pyqtSignal(int)  # Update maximum percent
    loadingDone = pyqtSignal()  # Finish loading

    def __init__(self, parent=None):
        super().__init__(parent)
        self.operation = 'Operation'
        self.i = 0
        self.interval = -1

    def set_interval(self, iter_num):
        """Set maximum number of operations

        :param iter_num: Number of operations
        """
        self.total = iter_num
        if iter_num <= 100:
            self.interval = 1
        else:
            self.interval = int(iter_num / 100)
        self.updateMaxPercent.emit(100)

    def check_percent(self, iter_):
        """Update percent for current operation number.
        Intended to be used after LoadingThread.set_interval

        :param iter_: 0 <= iter_ <= iter_num
        """
        if self.interval == 1:
            self.updatePercent.emit(int(iter_ / self.total * 100))
        elif self.interval < 0:
            return
        else:
            self.i += 1
            if self.i == self.interval:
                self.updatePercent.emit(int(iter_ / self.total * 100))
                self.i = 0

    def run(self, *args, **kwargs):
        raise NotImplementedError(f'{self.run} is not implemented')
