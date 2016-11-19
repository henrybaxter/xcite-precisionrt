import math
import numpy
import numbers
import json


from collada import Collada, source, geometry, material, scene


kelly_colors = [
    ("vivid_yellow", (255, 179, 0)),
    ("strong_purple", (128, 62, 117)),
    ("vivid_orange", (255, 104, 0)),
    ("very_light_blue", (166, 189, 215)),
    ("vivid_red", (193, 0, 32)),
    ("grayish_yellow", (206, 162, 98)),
    ("medium_gray", (129, 112, 102)),
    # these aren't good for people with defective color vision)
    ("vivid_green", (0, 125, 52)),
    ("strong_purplish_pink", (246, 118, 142)),
    ("strong_blue", (0, 83, 138)),
    ("strong_yellowish_pink", (255, 122, 92)),
    ("strong_violet", (83, 55, 122)),
    ("vivid_orange_yellow", (255, 142, 0)),
    ("strong_purplish_red", (179, 40, 81)),
    ("vivid_greenish_yellow", (244, 200, 0)),
    ("strong_reddish_brown", (127, 24, 13)),
    ("vivid_yellowish_green", (147, 170, 0)),
    ("deep_yellowish_brown", (89, 51, 21)),
    ("vivid_reddish_orange", (241, 58, 19)),
    ("dark_olive_green", (35, 44, 22)),
]

medium_colors = {
    'CU521xcom': (218, 138, 103),
    'VACUUM': (0, 0, 0),
    'BrassC35300_516kV': (181, 166, 66),
    'Air_516kV': (135, 206, 250),
    'BEAM': (44, 117, 255),
    'W_516kV': (120, 124, 13)
}


def calculate_vertices(d):
    vertices = []
    indices = []
    for cm in d['cms']:
        if cm['type'] == 'SLABS':
            xmin = ymin = cm['rmax_cm']
            xmax = ymax = -cm['rmax_cm']
            zmin = cm['zmin_slabs']
            for slab in cm['slabs']:
                zmax = zmin + slab['zthick']
                i = len(vertices) // 3
                vertices.extend([
                    xmax, ymax, zmin,
                    xmin, ymax, zmin,
                    xmin, ymin, zmin,
                    xmax, ymin, zmin,
                    xmin, ymin, zmax,
                    xmin, ymax, zmax,
                    xmax, ymax, zmax,
                    xmax, ymin, zmax,

                ])
                indices.extend([
                    # top
                    i, i + 1, i + 2,
                    i + 2, i + 3, i,
                    # right
                    i, i + 3, i + 4,
                    i + 4, i + 3, i + 7,
                    # front
                    i + 7, i + 3, i + 6,
                    i + 6, i + 3, i + 2,
                    # left
                    i + 2, i + 1, i + 5,
                    i + 5, i + 2, i + 6,
                    # bottom
                    i + 6, i + 5, i + 7,
                    i + 7, i + 5, i + 4,
                    # back
                    i + 4, i + 5, i + 1,
                    i + 1, i, i + 4
                ])
                zmin = zmax
    return vertices, indices


def export_dae(vertices, indices, fname):
    mesh = Collada()
    effect = material.Effect("effect0", [], "phong", diffuse=(1, 0, 0), specular=(0, 1, 0))
    mat = material.Material("material0", "mymaterial", effect)
    mesh.effects.append(effect)
    mesh.materials.append(mat)
    vert_src = source.FloatSource('verts-array', numpy.array(vertices), ('X', 'Y', 'Z'))
    inlist = source.InputList()
    inlist.addInput(0, 'VERTEX', '#verts-array')
    indices = numpy.array(indices)
    geom = geometry.Geometry(mesh, 'geometry0', 'linac', [vert_src])
    triset = geom.createTriangleSet(indices, inlist, "materialref")
    geom.primitives.append(triset)
    mesh.geometries.append(geom)
    matnode = scene.MaterialNode("materialref", mat, inputs=[])
    geomnode = scene.GeometryNode(geom, [matnode])
    node = scene.Node("node0", children=[geomnode])
    myscene = scene.Scene("myscene", [node])
    mesh.scenes.append(myscene)
    mesh.scene = myscene
    mesh.write(fname)


def medium_rgba(medium, alpha=0.6):
    rgb = medium_colors[medium]
    rgba = list(map(lambda v: v / 255, rgb)) + [alpha]
    return rgba


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


def cgs(d):
    lines = []
    cmax = 100
    #top = CGSElement()
    if d['isourc'] == '13':
        xmax = cmax
        xmin = 0
        ymax = d['ybeam']
        ymin = -d['ybeam']
        zmax = d['zbeam']
        zmin = -d['zbeam']
        xdeg = d['uinc'] * 180 / math.pi
        ydeg = d['vinc'] * 180 / math.pi
        #print(xdeg, ydeg)
        block = CGSBlock([xmin, xmax, ymin, ymax, zmin, zmax])
        #block.rotate(xdeg, [1, 0, 0])
        #block.rotate(ydeg, [0, 1, 0])
        block.color([0, 0, 0, .5])
        print(d['cms'][0]['zthick'])
        block.translate([0, 0, d['cms'][0]['zthick'] / 2])
        lines.append(block.render())

    z_min_cm = d['z_min_cm']
    for cm in d['cms']:
        if cm['type'] == 'SLABS':
            xmin = ymin = cm['rmax_cm']
            xmax = ymax = -cm['rmax_cm']
            zmin = max(z_min_cm, cm['zmin_slabs'])
            for i, slab in enumerate(cm['slabs']):
                zmax = zmin + slab['zthick']
                extents = [xmin, xmax, ymin, ymax, zmin, zmax]
                el = CGSElement()
                el.box(extents)
                el.color(medium_rgba(slab['medium']))
                lines.append(el.render())
                zmin = zmax
            z_min_cm = zmin
        elif cm['type'] == 'BLOCKS':
            pass
        elif cm['type'] == 'XTUBE':
            zmin = 0
            zmax = cmax
            ymin = -cm['rmax_cm']
            ymax = cm['rmax_cm']
            xmin = 0
            parent = CGSElement()
            for layer in cm['layers']:
                xmax = xmin + layer['dthick']
                extents = [xmin, xmax, ymin, ymax, zmin, zmax]
                block = CGSBlock(extents)
                block.color(medium_rgba(layer['medium']))
                parent.add(block)
                xmin = xmax
            # draw holder
            xmin = cm['rmax_cm']
            extents = [xmin, xmax, ymin, ymax, zmin, zmax]
            block = CGSBlock(extents)
            block.color(medium_rgba(cm['holder']['medium']))
            parent.add(block)
            
            parent.translate([0, 0, cm['zthick'] / 2])
            parent.mirror([-1, 0, 0])
            parent.rotate(cm['anglei'], [0, 1, 0])  # rotate in z
            parent.translate([0, 0, -cmax / 2])

            
            #parent.translate([0, 0, cm['zthick'] / 2])
            #parent.translate([0, 0, -(zmax - zmin) / 2])  # center in z
            # parent.translate(-(zmax - zmin) / 2)  # put it back
            intersection = CGSIntersection()
            intersection.add(parent)
            # block that is cmax in x, cmax in y, and from 0 to zthick in z
            extents = [-cmax, cmax, -cmax, cmax, 0, cm['zthick']]
            bounding = CGSBlock(extents)
            intersection.add(bounding)
            lines.append(intersection.render())
            z_min_cm = zmax
    return '\n'.join(lines)
