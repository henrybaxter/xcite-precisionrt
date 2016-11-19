from OCC.Display.SimpleGui import init_display
from OCC.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.GC import GC_MakeSegment
from OCC.BRepBuilderAPI import BRepBuilderAPI_Transform, BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeEdge
from OCC.gp import gp, gp_Pnt, gp_Vec, gp_Dir, gp_Trsf, gp_Ax1
from PyQt5.QtCore import QTimer
from OCC.GC import GC_MakeLine

display, start_display, add_menu, add_function_to_menu = init_display('qt-pyqt5', (1024,768))
my_box = BRepPrimAPI_MakeBox(10., 20., 30.).Shape()
pnt1 = gp_Pnt(-1, -1, 40)
pnt2 = gp_Pnt(-1, -10, 40)

origin = gp_Pnt(0, 0, 0)
xdirection = gp_Dir(1, 0, 0)
xaxis = gp.OX()
t = gp_Trsf()
t.SetMirror(xaxis)
xline = GC_MakeLine(gp.OX()).Value()

geom_curve = GC_MakeSegment(pnt1, pnt2).Value()
bt = BRepBuilderAPI_Transform(my_box, t)
# what about a geom line!?
# ok what about something rotated?
#edge = BRepBuilderAPI_MakeEdge(10)
#my_box2 = BRepPrimAPI_MakeBox(5., 5., 5.).Shape()
#wire = BRepBuilderAPI_MakeWire(10, 10).Shape()

def f():
    display.DisplayShape(geom_curve, update=True)
    display.DisplayShape(bt.Shape(), update=True)
QTimer().singleShot(300, f)
display.DisplayShape(GC_MakeLine(gp.OX()).Value())
display.DisplayShape(GC_MakeLine(gp.OY()).Value())
display.DisplayShape(GC_MakeLine(gp.OZ()).Value())
display.DisplayShape(my_box, update=True)
#display.DisplayShape(wire, update=True)

display.Rotation(10, 100)
start_display()

# ok what if we change it?
