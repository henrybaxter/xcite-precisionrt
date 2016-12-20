import statistics

import numpy as np

import egsinp


def polygon_area(corners):
    n = len(corners)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = abs(area) / 2.0
    return area


def block_stats(block):
    max_x = 0
    max_y = 0
    min_x = 0
    min_y = 0
    areas = []
    for region in block['regions']:
        area = polygon_area([(p['x'], p['y']) for p in region['points']])
        max_x = max(max_x, max(p['x'] for p in region['points']))
        max_y = max(max_y, max(p['y'] for p in region['points']))
        min_x = min(min_x, min(p['x'] for p in region['points']))
        min_y = min(min_y, min(p['y'] for p in region['points']))
        areas.append(area)
    return {
        'regions': len(areas),
        'total_area': sum(areas),
        'average_region_area': statistics.mean(areas),
        'max_x': max_x,
        'max_y': max_y,
        'min_x': min_x,
        'min_y': min_y,
        'z_focus': block['zfocus']
    }


def analyze(collimator):
    blocks = [block for block in collimator['cms'] if block['type'] == 'BLOCK']
    stats = [block_stats(block) for block in blocks]
    return {
        'blocks': stats,
        'total_blocks': len(blocks),
        'anode_area': stats[0]['total_area'],
        'exit_area': stats[-1]['total_area'],
        'length': blocks[-1]['zmax'] - blocks[0]['zmin'],
        'width': np.array([[b['max_x'], b['max_y']] for b in stats]).max() * 2
    }


if __name__ == '__main__':
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    args = parser.parse_args()
    if not args.input.endswith('.egsinp'):
        args.input += '.egsinp'
    stats = analyze(egsinp.parse_egsinp(open(args.input).read()))
    print(json.dumps(stats, sort_keys=True, indent=2))
