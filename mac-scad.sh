/Applications/OpenSCAD.app/Contents/MacOS/OpenSCAD $1.scad --camera=-0,3.60,4.92,90.0,0.0,0.0,62.09 -o $1_side.png --imgsize=1600,600 --autocenter && open $1_side.png
/Applications/OpenSCAD.app/Contents/MacOS/OpenSCAD $1.scad --camera=-0,0,5,0.0,0.0,0.0,12 -o $1_phantom.png --imgsize=1600,600 --autocenter && open $1_phantom.png
/Applications/OpenSCAD.app/Contents/MacOS/OpenSCAD $1.scad --camera=-0,0,5,180.0,0.0,0.0,6 -o $1_anode.png --imgsize=1600,600 --autocenter && open $1_anode.png
