from PyQt5.QtWidgets import QApplication
import sys
from ui import OpenDialog


if __name__ == "__main__":
    app = QApplication(sys.argv)
    dialog = OpenDialog()
    dialog.show()

    sys.exit(app.exec_())
