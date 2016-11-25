"""
Create gap regions using the source plane, and a zfocus from
the phantom plane.
"""
import math

def read(path):
    """Returns list of (width, gap) including center hole of stamped collimator,
    from center to one extreme."""
    result = []
    for line in open(path):
        width, gap, edge = [float(e) / 10 for e in line.split(',')]
        # we can avoid edge length since we know height and width of paralellogram
        result.append((width, gap))
    result.reverse()
    return result


def region_points(x, y, width, xmin, xmax):
    """Returns points counterclockwise from maximum in y direction that form
    parallelogram region with given (x, y) center, width, and (xmin, xmax).

    Note (xmin, xmax) are absolute, not relative to the x coordinate.
    """
    right = (x, y + width / 2)
    top = (xmax, y)
    left = (x, y - width / 2)
    bottom = (xmin, y)
    return (right, top, left, bottom)


def centers_and_regions(data, x, y, xmin, xmax):
    """
    Starting at (x, y), using extents (xmin, xmax),
    use widths and gaps to generate regions.
    """
    # starting at (x, y), using extents xmin
    # first hole is not duplicated, so it is treated differently
    width, gap = data[0]
    result = []
    result.append(((x, y), region_points(x, y, width, xmin, xmax)))
    y = width / 2 + gap
    for width, gap in data[1:]:
        y += width / 2
        result.append(((x, y), region_points(x, y, width, xmin, xmax)))
        result.insert(0, ((x, -y), region_points(x, -y, width, xmin, xmax)))
        y += width / 2 + gap
    return result


def at_z(point1, point2, z):
    """
    >>> at_z((0, 1, 1), (0, 3, 3), 2)
    (0.0, 2.0)
    >>> at_z((0, 1, 0), (0, 3, 2), 1)
    (0.0, 2.0)
    >>> at_z((0, 3, 3), (0, 1, 1), 2)
    (0.0, 2.0)
    >>> at_z((0, 1, 1), (0, 1, 1), 0)
    Traceback (most recent call last):
    ...
    AssertionError: z must be between points
    """
    x1, y1, z1 = point1
    x2, y2, z2 = point2
    assert z1 <= z <= z2 or z2 <= z <= z1, "z must be between points"
    y = (y2 - y1) / (z2 - z1) * (z - z1) + y1
    x = (x2 - x1) / (z2 - z1) * (z - z1) + x1
    return (x, y)


def make_blocks(collimator_length, n_blocks):
    xmax = math.sqrt(2) / 10  # should be in centimeters
    xmin = -xmax
    x = 0
    y = 0
    phantom = centers_and_regions(read('data/phantom.csv'), x, y, xmin, xmax)
    source = centers_and_regions(read('data/source.csv'), x, y, xmin, xmax)
    blocks = []
    for i in range(n_blocks):
        zmin = collimator_length / n_blocks * i
        zmax = collimator_length / n_blocks * (i + 1)
        print(zmin, zmax)
        block = {
            'regions': [],
            'zmin': zmin,
            'zmax': zmax,
            'xpmax': 100,
            'ypmax': 100,
            'xnmax': -100,
            'ynmax': -100
        }
        for (_, s_region), (_, p_region) in zip(source, phantom):
            region = []
            for (s_x, s_y), (p_x, p_y) in zip(s_region, p_region):
                region.append(at_z((s_x, s_y, 0), (p_x, p_y, collimator_length), zmax))
            block['regions'].append(region)
        blocks.append(block)
    return blocks


def make_hblocks(**kwargs):
    length = kwargs.get('length', 12)
    septa = kwargs.get('septa', .05)
    width = kwargs.get('width', 50)  # phantom width
    target_distance = kwargs.get('target_distance', 75)  # focus from phantom side
    target_width = kwargs.get('target_width', 1)
    n_blocks = kwargs.get('blocks', 10)
    size = kwargs.get('size', .2)

    # source is considered z = 0, positive 'down'

    # first phantom hole on the right (to be copied, translated, and reflected)
    # z of these is length
    points = []
    # center = ((septa / 2 + size, 0))
    points = [
        (septa / 2, 0),
        (septa / 2 + size / 2, -size / 2),
        (septa / 2 + 3 * size / 2, -size / 2),
        (septa / 2 + size * 2, 0),
        (septa / 2 + 3 * size / 2, size / 2),
        (septa / 2 + size / 2, size / 2)
    ]
    dx = septa / 2 + size * 2 + septa / 2
    width_remaining = width / 2 - dx
    regions = []

    def translate(points, dx):
        translated = []
        for x, y in points:
            translated.append((x + dx, y))
        return translated
    i = 0
    while width_remaining >= 0:
        translated = translate(points, i * dx)
        regions.append(translated)
        width_remaining -= dx
        i += 1
    # here we copy/reflect
    def reflect(points):
        reflected = []
        for x, y in points:
            reflected.append((-x, y))
        return reflected
    # ok now we have all points on phantom side
    # and we have the target points
    # we need to choose our x2 value
    def find_x(x0, z0, x1, z1, z2):
        m = (z1 - z0) / ((x1 - x0) or 0.000001)
        x2 = (z2 - z0) / m + x0
        return x2
    # so now we need to interpolate between these
    # but for now, do we care? can't we just run it?
    # no, we need them, and a lot of them probably
    # so how do we interpolate?
    # we have all the value, and also what z focus??
    # z focus will be negative infty

    block_length = length / n_blocks
    target_x_left = -target_width / 2
    target_x_right = target_width / 2
    target_z = length + target_distance
    blocks = []
    for i in range(n_blocks):
        current_z = i * block_length
        block_regions = []
        for region in regions:
            phantom_x_left = region[0][0]
            left_x = find_x(target_x_left, target_z, phantom_x_left, length, current_z)
            phantom_x_right = region[3][0]
            right_x = find_x(target_x_right, target_z, phantom_x_right, length, current_z)
            size = right_x - left_x
            # print(size)
            block_regions.append([
                (left_x, 0),
                (left_x + size / 2, -size / 2),
                (left_x + 3 * size / 2, -size / 2),
                (left_x + 2 * size, 0),
                (left_x + 3 * size / 2, size / 2),
                (left_x + size / 2, size / 2)
            ])
        for region in block_regions[:]:
            block_regions.insert(0, reflect(region))
        blocks.append({
            'zmin': current_z,
            'zmax': current_z + block_length,
            'regions': block_regions
        })
    return blocks
    # we want the center point to move
    # and then we want the same hole?
    # no we want to take
    # we need some divergence going on right?
    # we use a z focus that opens us up
    #
    # assume septa width separation and endpoints
    # and assume size is square width, and then we have half square width more each side
    # so size * 2 is the 'length' of a cell
    # then we make it diverge from there?
    # we take the center point of have it converge. then we work back for the divergence
    # the divergence is found by taking that point, then going backwards
    # and we start from the center, and build using half septa width

    # septa means nothing with hexagons right? but let's pretend
    # ho

    # make these 6 pointed regions
    # make them


def area_polygon(corners):
    n = len(corners)  # of corners
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = abs(area) / 2.0
    return area

if __name__ == '__main__':
    blocks = make_hblocks(interpolating_blocks=10)
    for i in range(10):
        print(blocks[i]['regions'][0][0], blocks[i]['regions'][0][3])
    #print(blocks[0]['regions'][1])
    #print(blocks[0]['regions'][0])
    print('n blocks', len(blocks))
    print('n regions', len(blocks[0]))

    """
    blocks = make_blocks(12, 10)
    areas = []
    for region in blocks[-1]['regions']:
        areas.append(area_polygon(region))
    area_bottom = statistics.mean(areas)
    areas = []
    for region in blocks[0]['regions']:
        areas.append(area_polygon(region))
    area_top = statistics.mean(areas)
    print(area_top / area_bottom)
    # print(blocks[0][0])
    # data = withzfocus()
    # find change in size
    import doctest
    doctest.testmod()
    # print(data)
    print(blocks[0]['regions'][0])
    print(blocks[-1]['regions'][0])
    r1 = blocks[0]['regions'][0]
    r2 = blocks[-1]['regions'][0]
    d1 = r1[0][1] - r1[2][1]
    d2 = r2[0][1] - r2[2][1]
    print(d1, d2)
    """
