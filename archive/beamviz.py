import sys
import logging

import yaml
import numpy as numpy

from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout
from PyQt5.Qsci import QsciScintilla, QsciLexerYAML
from PyQt5.QtGui import QFont, QFontInfo, QFontDatabase, QFontMetrics, QColor
from PyQt5.QtCore import QSettings
import pyqtgraph as pg
from pyqtgraph.opengl import GLViewWidget, GLGridItem

logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)


def getFont():
    preferredFamilies = [
        'Inconsolata', 'Consolas', 'Deja Vu Sans Mono', 'Droid Sans Mono',
        'Proggy', 'Monofur', 'Profont', 'Monaco', 'Andale Mono', 'Courier'
    ]
    font = QFont()
    for family in preferredFamilies:
        logger.debug("Attempting to choose {}".format(family))
        font = QFont()
        font.setFamily('Consolas')
        fontInfo = QFontInfo(font)
        if fontInfo.family() == family:
            logger.info("Succeeded in choosing family {}".format(family))
            break
    font.setPointSize(14)
    return font


def getMarginWidth(font):
    fontMetrics = QFontMetrics(font)
    return fontMetrics.width('0000') + 6


def getEditor():
    editor = QsciScintilla()
    lexer = QsciLexerYAML()
    editor.setLexer(lexer)
    font = getFont()
    # lexer.setDefaultFont(font)
    lexer.setFont(font)
    settings = QSettings()
    lexer.readSettings(settings)
    print(settings.allKeys())
    lexer.setDefaultPaper(QColor('#252721'))
    for i in range(10):
        lexer.setPaper(QColor('#252721'), i)
        lexer.setColor(QColor('#f8f8f2'), i)
        print(lexer.color(i).name())
    lexer.setColor(QColor('#e6db74'), 0)  # foreground (yellow)
    lexer.setColor(QColor('#75715e'), 1)  # comment (gray)
    lexer.setColor(QColor('#f92672'), 2)  # identifier (red)
    lexer.setColor(QColor('#ae81ff'), 3)  # keyword (purple)
    lexer.setColor(QColor('#ae81ff'), 4)  # number (purple)
    lexer.setColor(QColor('#ae81ff'), 5)  # reference (purple)
    lexer.setColor(QColor('#ae81ff'), 6)  # documentdelimiter (purple)
    lexer.setColor(QColor('#ae81ff'), 7)  # text block marker (purple)
    lexer.setColor(QColor('#f92672'), 8)  # syntax error marker (red)
    lexer.setColor(QColor('#f92672'), 9)  # operator (red)
    # editor.setFont(font)
    # editor.setMarginsFont(font)
    editor.setCaretForegroundColor(QColor('#f8f8f0'))
    editor.setMarginWidth(0, getMarginWidth(font))
    editor.setMarginWidth(1, 0)
    editor.setMarginLineNumbers(0, True)
    editor.setMarginsBackgroundColor(QColor('#252721'))
    editor.setMarginsForegroundColor(QColor('#f8f8f2'))
    editor.setExtraAscent(3)
    editor.setTabWidth(4)
    editor.setMinimumSize(600, 450)
    return editor


def getViewer():
    viewer = GLViewWidget()
    viewer.setMinimumSize(500, 500)
    return viewer


def getLayout():
    grid = QGridLayout()
    grid.addWidget(getEditor())
    grid.addWidget(getViewer(), 0, 1)  # second column
    grid.setSpacing(0)
    grid.setContentsMargins(0, 0, 0, 0)
    return grid


def getWindow():
    window = QWidget()
    window.setLayout(getLayout())
    return window

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = getWindow()
    window.show()
    sys.exit(app.exec_())
