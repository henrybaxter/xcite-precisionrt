#!/usr/bin/env python

# Copyright 2009-2016 Thomas Paviot (tpaviot@gmail.com)
##
# This file is part of pythonOCC.
##
# pythonOCC is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
##
# pythonOCC is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
##
# You should have received a copy of the GNU Lesser General Public License
# along with pythonOCC.  If not, see <http://www.gnu.org/licenses/>.

import logging
import os
import sys

from OCC import VERSION

from PyQt5 import QtWidgets

from qtDisplay import qtViewer3d

log = logging.getLogger(__name__)


def check_callable(_callable):
    assert callable(_callable), 'the function supplied is not callable'


def init_display(backend_str=None, size=(1024, 768)):
    """ This function loads and initialize a GUI using either wx, pyq4, pyqt5 or pyside.
    If ever the environment variable PYTHONOCC_SHUNT_GUI, then the GUI is simply ignored.
    It can be useful to test some algorithms without being polluted by GUI statements.
    This feature is used for running the examples suite as tests for
    pythonocc-core development.
    """

    class MainWindow(QtWidgets.QMainWindow):

        def __init__(self, *args):
            QtWidgets.QMainWindow.__init__(self, *args)
            self.canva = qtViewer3d(self)
            self.setWindowTitle("pythonOCC-%s 3d viewer ('%s' backend)" % (VERSION, 'qt-pyqt5'))
            self.resize(size[0], size[1])
            self.setCentralWidget(self.canva)
            if not sys.platform == 'darwin':
                self.menu_bar = self.menuBar()
            else:
                # create a parentless menubar
                # see: http://stackoverflow.com/questions/11375176/qmenubar-and-qmenu-doesnt-show-in-mac-os-x?lq=1
                # noticeable is that the menu ( alas ) is created in the
                # topleft of the screen, just
                # next to the apple icon
                # still does ugly things like showing the "Python" menu in
                # bold
                self.menu_bar = QtWidgets.QMenuBar()
            self._menus = {}
            self._menu_methods = {}
            # place the window in the center of the screen, at half the
            # screen size
            self.centerOnScreen()

        def centerOnScreen(self):
            '''Centers the window on the screen.'''
            resolution = QtWidgets.QDesktopWidget().screenGeometry()
            print(resolution)
            print(self.frameSize())
            self.move((resolution.width() / 2) - (self.frameSize().width() / 2),
                      (resolution.height() / 2) - (self.frameSize().height() / 2))

        def add_menu(self, menu_name):
            _menu = self.menu_bar.addMenu("&" + menu_name)
            self._menus[menu_name] = _menu

        def add_function_to_menu(self, menu_name, _callable):
            check_callable(_callable)
            try:
                _action = QtWidgets.QAction(_callable.__name__.replace('_', ' ').lower(), self)
                # if not, the "exit" action is now shown...
                _action.setMenuRole(QtWidgets.QAction.NoRole)
                _action.triggered.connect(_callable)

                self._menus[menu_name].addAction(_action)
            except KeyError:
                raise ValueError('the menu item %s does not exist' % menu_name)

    # following couple of lines is a twek to enable ipython --gui='qt'
    app = QtWidgets.QApplication.instance()  # checks if QApplication already exists

    if not app:  # create QApplication if it doesnt exist
        app = QtWidgets.QApplication(sys.argv)
    #app.setGraphicsSystem('native')
    win = MainWindow()
    #win.setFixedSize(300, 300)
    win.show()
    win.canva.InitDriver()
    display = win.canva._display
    if sys.platform != "linux2":
        display.EnableAntiAliasing()
    # background gradient
    display.set_bg_gradient_color(206, 215, 222, 128, 128, 128)
    # display black trihedron
    display.display_trihedron()

    def add_menu(*args, **kwargs):
        win.add_menu(*args, **kwargs)

    def add_function_to_menu(*args, **kwargs):
        win.add_function_to_menu(*args, **kwargs)

    def start_display():
        win.raise_()  # make the application float to the top
        app.exec_()
    return display, start_display, add_menu, add_function_to_menu


if __name__ == '__main__':
    display, start_display, add_menu, add_function_to_menu = init_display("qt-pyqt5")
    from OCC.BRepPrimAPI import BRepPrimAPI_MakeSphere, BRepPrimAPI_MakeBox

    def sphere(event=None):
        display.DisplayShape(BRepPrimAPI_MakeSphere(100).Shape(), update=True)

    def cube(event=None):
        display.DisplayShape(BRepPrimAPI_MakeBox(1, 1, 1).Shape(), update=True)

    def exit(event=None):
        sys.exit()

    cube()
    add_menu('primitives')
    add_function_to_menu('primitives', sphere)
    add_function_to_menu('primitives', cube)
    add_function_to_menu('primitives', exit)
    start_display()
