import statistics
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


def collimator_stats(blocks):
    return {
        'blocks': [block_stats(block) for block in blocks],
        'total_blocks': len(blocks),
        'anode_area': blocks[0]['total_area'],
        'exit_area': blocks[-1]['total_area']
    }


if __name__ == '__main__':
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    args = parser.parse_args()
    cms = egsinp.parse_egsinp(open(args.input).read())['cms']
    blocks = [cm for cm in cms if cm['type'] == 'BLOCK']
    stats = collimator_stats(blocks)
    print(json.dumps(stats, sort_keys=True, indent=2))
