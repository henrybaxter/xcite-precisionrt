import logging
import numpy
import math
import sys
import argparse
from pyparsing import (Keyword, ZeroOrMore, Regex, Optional, Word,
                       Suppress, Group, alphanums)

from collada import Collada, source, geometry, material, scene


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def build_grammar():
    sep = Suppress(',')
    floatNumber = Regex(r'-?\d+(\.\d*)?([eE]\d+)?')
    radius = floatNumber
    coord = floatNumber + sep + floatNumber + sep + floatNumber
    quad = Keyword('1QUAD') + coord + sep + coord + sep + coord + sep + coord
    disk = Keyword('3DISK') + coord + sep + radius
    line = (disk | quad | Word(alphanums)) + Optional(sep)
    return ZeroOrMore(Group(line))


# args
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('egsgeom')
    parser.add_argument('dae', nargs='?', default='output.dae')
    return parser.parse_args()


def read_egs(fname):
    try:
        f = open(fname)
    except OSError as e:
        print("Could not open {}: {}".format(fname, str(e)))
        sys.exit(1)
    return f.read()


def calculate_verticies(entities):
    vertices = []
    indices = []
    for entity in entities:
        if entity[0] == '1QUAD':
            i = len(vertices) // 3
            vertices.extend(map(float, entity[1:]))
            indices.extend([i, i + 1, i + 2, i + 2, i + 3, i])
        elif entity[0] == '3DISK':
            # assume z is the axis of the disk
            x, y, z, r = map(float, entity[1:])
            n_triangles = 12
            multiplier = 2 * math.pi / n_triangles
            for i in range(n_triangles):
                a1 = multiplier * i
                a2 = a1 + multiplier
                ind = len(vertices) // 3
                vertices.extend([x, y, z])
                vertices.extend([x + r * math.cos(a1), y + r * math.sin(a1), z])
                vertices.extend([x + r * math.cos(a2), y + r * math.sin(a2), z])
                indices.extend([ind, ind + 1, ind + 2])
        else:
            raise ValueError("Unsupported primitive {}".format(entity[0]))
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

if __name__ == '__main__':
    args = parse_args()
    egsgeom = read_egs(args.egsgeom)
    grammar = build_grammar()
    result = grammar.parseString(egsgeom)
    vertices, indices = calculate_verticies(result)
    export_dae(vertices, indices, args.dae)
