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
                region.append(at_z((s_x, s_y, 0), (p_x, p_y, collimator_length), zmin))
            block['regions'].append(region)
        blocks.append(block)
    return blocks

if __name__ == '__main__':
    blocks = make_blocks(12, 1, 0.1)
    for block in blocks:
        print(block)
        sys.exit()
    # print(blocks[0][0])
    # data = withzfocus()
    import doctest
    doctest.testmod()
    # print(data)
