import numbers
import json
from math import atan, sqrt
from scipy.optimize import fsolve

from interpolation import


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

    def zbox(self, region, zfocus):
        points = [(x, y, 0) for x, y in region]
        points.append((0, 0, zfocus))
        # one that makes region
        faces = [list(range(len(region)))]
        # then n-1 that make triangles
        last = len(region)
        for i in range(len(region)):
            faces.append((i, (i + 1) % len(region), last))
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

if __name__ == '__main__':
    phantom_focus_radius = 5
    phantom_focus_color = [1.0, 0, 0, 1.0]
    phantom_skin_radius = 100
    phantom_skin_color = [0, 0, 1.0, 0.1]
    # l_t = length_target = 750
    b = beam_width = 2
    s_w = septa_width = .140
    s_m = septa_minimum = .039
    d_sf = distance_skin_focus = 201
    d_tc = distance_target_collimator = 516
    d_cs = distance_collimator_skin = 400
    l_c = length_collimator = 122
    d_cf = distance_collimator_focus = d_cs + d_sf
    d_tf = distance_target_focus = d_tc + l_c + d_cs + d_sf
    """
    # tan(theta) = (l_t / 2) / d_tf
    theta = atan((l_t / 2 / d_tf))
    # 2 * d_cf * tan(theta)

    d = d_cf + l_c
    c_s_xmax = 686 / 2  # (d_cf + l_c) * ((l_t / 2) / d_tf)
    c_p_xmax = 565 / 2  # d_cf * ((l_t / 2) / d_tf)
    # calculate linear fit on data
    # alternatively could replace functions below with quadratics
    # that fit the data we have a bit better
    # this one is easier to tweak though :)
    b_s_max = 2.222
    b_s_min = 2.000
    b_p_max = 2.222
    b_p_min = 1.975
    s_slope = (b_s_max - b_s_min) / c_s_xmax
    p_slope = (b_p_max - b_p_min) / c_p_xmax
    s_m_s_max = 1.376
    s_m_s_min = 0.662
    s_m_p_max = 0.663
    s_m_p_min = 0.000
    s_m_s_slope = (s_m_s_max - s_m_s_min) / c_s_xmax
    s_m_p_slope = (s_m_p_max - s_m_p_min) / c_p_xmax

    def source_width(x):
        return s_slope * x + b_s_min

    # def source_width(x):
    #     return 3.43e-05 * x * x - 0.0055 * x + 2.222

    def phantom_width(x):
        return p_slope * x + b_p_min

    # def phantom_width(x):
    #     return 2.500e-05 * x * x - 0.005 * x + 2.222

    def source_space(x):
        return s_m_s_slope * x + s_m_s_min

    def phantom_space(x):
        return s_m_p_slope * x + s_m_p_min

    def source_total(x):
        w = source_width(x)
        h = 2 * sqrt(pow(w / 2, 2) + pow(w / 2, 2))
        # squares are rotated!
        return h + source_space(x)

    def phantom_total(x):
        w = phantom_width(x)
        h = 2 * sqrt(pow(w / 2, 2) + pow(w / 2, 2))
        # squares are rotated!
        return h + phantom_space(x)

    def next_s_x(x):
        return x_prev + source_total(x_prev) / 2 + source_total(x) / 2 - x

    def next_p_x(x):
        return x_prev + phantom_total(x_prev) / 2 + phantom_total(x) / 2 - x

    s_centers = []
    x_prev = 0
    while x_prev < c_s_xmax:
        s_centers.append(x_prev)
        x_prev = fsolve(next_s_x, x_prev)[0]

    p_centers = []
    x_prev = 0
    while x_prev < c_p_xmax:
        p_centers.append(x_prev)
        x_prev = fsolve(next_p_x, x_prev)[0]

    print('source centers', len(s_centers), s_centers)
    print('phantom centers', len(p_centers), p_centers)

    z_focuses = []
    ZMAX = 160
    for s_y, p_y in zip(s_centers, p_centers):
        # here we have to translate everything for fuck's sake.
        s_z = d_tc
        p_z = d_tc + l_c
        if p_y - s_y == 0:
            # infinity
            z_focuses.append(ZMAX)
        else:
            m = (p_z - s_z) / (p_y - s_y)
            z = m * p_y + p_z
            z_focuses.append(z)

    print('z focuses', z_focuses)

    # now we need to construct the squares on the SOURCE side
    # which are then rotated, translated, and built into zblocks
    zblocks = CGSElement()
    for s_y, z in zip(s_centers, z_focuses):
        w = source_width(s_y)
        # septa width is used, NOT spacing, for construction
        s_m
        x1 = x2 = w / 2
        x3 = x4 = - w / 2
        y1 = y4 = s_y + w / 2
        y2 = y3 = s_y - w / 2
        region = [
            (x1, y1),
            (x2, y2),
            (x3, y3),
            (x4, y4)
        ]
        zblocks.add(CGSZBlock(region, z))
    xmin = -50
    xmax = 50
    ymin = -c_s_xmax
    ymax = c_s_xmax
    zmin = 0
    zmax = l_c
    extents = xmin, xmax, ymin, ymax, zmin, zmax
    block = CGSBlock(extents)
    collimator_i = CGSIntersection()
    collimator_i.add(block)
    collimator_i.add(zblocks)
    collimator = CGSElement()
    collimator.add(collimator_i)
    collimator.translate([0, 0, d_tc])
    """

    # phantom
    phantom_focus = CGSSphere(phantom_focus_radius)
    phantom_focus.color(phantom_focus_color)
    phantom_skin = CGSSphere(phantom_skin_radius)
    phantom_skin.color(phantom_skin_color)
    phantom = CGSElement()
    phantom.add(phantom_focus)
    phantom.add(phantom_skin)
    phantom.translate([0, 0, d_tf])

    # target
    xmin = -5
    xmax = 5
    ymin = -375
    ymax = 375
    zmin = -1
    zmax = 0
    extents = xmin, xmax, ymin, ymax, zmin, zmax
    target = CGSBlock(extents)

    """
    # collimator
    # need a function for the centerpoints of the holes
    # region = [(10, 10), (-10, 10), (-10, -10), (10, -10)]
    # zbox = CGSZBlock(region, d_tf)
    # zboxes =
    # for i in range()
    # collimator.add(zbox)
    """

    # y_i = lambda i: i * sqrt(2 * b) + s_m
    scene = CGSElement()
    scene.add(phantom)
    scene.add(target)
    # scene.add(collimator)
    open('out.scad', 'w').write(scene.render())
