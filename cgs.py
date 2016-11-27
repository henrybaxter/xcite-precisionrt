import numbers
import json

"""
Wrapper to help write OpenSCAD scripts
"""


class CGSElement(object):

    def __init__(self):
        self.children = []
        self.primitives = []
        self.transformations = []

    def add(self, child):
        self.children.append(child)

    def box(self, extents):
        assert len(extents) == 6
        xmin, xmax, ymin, ymax, zmin, zmax = extents
        points = [
            (xmax, ymax, zmin),
            (xmin, ymax, zmin),
            (xmin, ymin, zmin),
            (xmax, ymin, zmin),
            (xmax, ymax, zmax),
            (xmin, ymax, zmax),
            (xmin, ymin, zmax),
            (xmax, ymin, zmax),
        ]
        faces = [
            [0, 1, 2, 3],
            [3, 2, 6, 7],
            [2, 1, 5, 6],
            [0, 3, 7, 4],
            [1, 0, 4, 5],
            [7, 6, 5, 4]
        ]
        self.primitives.append(('polyhedron', points, faces))

    def sphere(self, radius):
        assert radius > 0
        self.primitives.append(('sphere', radius))

    def color(self, rgba):
        assert len(rgba) == 4
        self.transformations.append(('color', rgba))

    def mirror(self, normal):
        assert len(normal) == 3
        self.transformations.append(('mirror', normal))

    def rotate(self, degree, axis):
        assert -360 <= degree <= 360
        assert len(axis) == 3
        self.transformations.append(('rotate', degree, axis))

    def translate(self, vector):
        self.transformations.append(('translate', vector))

    def render(self, level=0):
        rendered = []
        padding = level * '\t'
        for transformation in self.transformations:
            rendered.append(padding + self.stringify(transformation))
        rendered.append(padding + '{')
        for primitive in self.primitives:
            rendered.append(padding + '\t' + self.stringify(primitive) + ';')
        for child in self.children:
            rendered.append(child.render(level + 1))
        rendered.append(padding + '}')
        return '\n'.join(rendered)

    def stringify(self, command):
        call, *args = command
        args = [
            str(arg) if isinstance(arg, numbers.Number) else json.dumps(arg)
            for arg in args
        ]
        return '{}({})'.format(call, ', '.join(args))


class CGSIntersection(object):
    def __init__(self):
        self.elements = []

    def add(self, el):
        self.elements.append(el)

    def render(self, level=0):
        rendered = []
        for el in self.elements:
            rendered.append(el.render())
        return 'intersection() {' + '\n'.join(rendered) + '}'


class CGSBlock(CGSElement):
    def __init__(self, extents):
        super().__init__()
        self.box(extents)


class CGSSphere(CGSElement):
    def __init__(self, radius):
        super().__init__()
        self.sphere(radius)


class CGSZBlock(CGSElement):
    def __init__(self, region, zfocus):
        super().__init__()
        self.zbox(region, zfocus)
