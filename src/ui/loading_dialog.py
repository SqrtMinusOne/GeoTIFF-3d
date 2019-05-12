from ui_compiled.loadingdialog import Ui_LoadingDialog
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QDialog
from datetime import datetime
from loading_wrapper import LoadingThread


__all__ = ['LoadingDialog', 'LoadingWrapper']


def format_time(time):
    return f"{time.hour:02}:{time.minute:02}:{time.second:02}"


def format_delta(delta):
    totsec = delta.total_seconds()
    h = int(totsec // 3600)
    m = int((totsec % 3600) // 60)
    sec = int((totsec % 3600) % 60)
    return f"{h:02}:{m:02}:{sec:02}"


class LoadingDialog(QDialog, Ui_LoadingDialog):
    def __init__(self, operation: str, total=0, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.total = total
        self.progressBar.setMaximum(total)
        self.operationLabel.setText(operation)
        self.statusDescLabel.hide()
        self.statusLabel.hide()

    def show(self):
        super().show()
        self.start_time = datetime.now()
        self.startTimeShow.setText(format_time(self.start_time))

    def set_status(self, status):
        self.statusLabel.show()
        self.statusDescLabel.show()
        self.statusLabel.setText(status)

    def set_operation(self, operation):
        self.operationLabel.setText(operation)

    def set_done(self, value: int):
        if value > 0:
            self.progressBar.setValue(value)
            delta = datetime.now() - self.start_time
            op_time = delta / value
            # remaining = (self.total - value) * op_time
            remaining = (100 - value) * op_time
            self.etaShow.setText(format_delta(remaining))

    def ready(self):
        self.hide()


class LoadingWrapper(QObject):
    loadingDone = pyqtSignal()

    def __init__(self, thread: LoadingThread, parent=None):
        super().__init__(parent)
        self.thread = thread
        operations = 100 if thread.interval > 0 else 0
        self.dialog = LoadingDialog(thread.operation, operations)
        thread.updateStatus.connect(self.dialog.set_status)
        thread.updatePercent.connect(self.dialog.set_done)
        thread.loadingDone.connect(self.dialog.ready)
        thread.loadingDone.connect(lambda: self.loadingDone.emit())
        thread.updateMaxPercent.connect(self.dialog.progressBar.setMaximum)

    def start(self):
        self.dialog.show()
        self.thread.start()
