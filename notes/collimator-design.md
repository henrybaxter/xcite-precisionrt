- always use regular sized hexagonal holes where we specify the inradius of the hexagon.
- always arrange the hexagons so their sides correspond
- allow for multiple rows (just build using beam height and septa y)
- choose a distribution along x and along y
	- for each source, we have the points right. now the question is, how do we map these points onto the target
	- we choose a 
	- since they are regular, we make the assumption that we want to 
	- a point
	- all
	- left/right segments
	- proportional segments 

we also have a septa gap at at the anode side.

divergence along y and divergence along x

[septa.x]
[septa.y]
[inradius.x]
[inradius.y]

etc


to a point. to a circle.

to a point is simple.

to a circle is more interesting.

but what if we chose, not a circle, but some kind of combination?

same shape but transformed in some way? scaled in some way?

ok so the point that we choose is a function of the center point of the source hole.

the center point has x and y coordinates, and the point we choose does as well.

from there we can choose to map to a circle the size of the target, or we can choose something else.

or, instead of a circle, we could map to something else, such as a line, or the same shape of hexagon scaled in some way


so we have:

i had:
[lesion]
distance = 40.0
diameter = 1.0

i had:
[target.mapto.rectangle]
height = 0.0
width = 1.0
[target.distribution.center]

they had:
[target.mapto.point]
[target.distribution.center]

magdalena had:
[target.mapto.circle]
diameter = 1.0
[target.distribution.center]

i believe in:
[target.mapto.rectangle]
y = 0
x = 0.5
[target.distribution.left-right]

or better yet:
[target.mapto.chords]
x = 0.5
[target.distribution.left-right]


or possibly:
[target.mapto.chords]
x = 0.1
[target.distribution.polynomial]
x = [0.5, 0.0, 1.0]
