import sys

from PyQt5.QtWidgets import QApplication, QWidget

from OCC.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Display import OCCViewer

class View(QWidget):



if __name__ == '__main__':
    app = QApplication(sys.argv)
    view = View()
    view.setWindowTitle("Basic OCC Viewer")
    view.resize(1024, 768)
    view.show()
    sys.exit(app.exec_())
