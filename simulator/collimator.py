import sys
import os
import toml
import subprocess
import argparse
import copy

import numpy as np
from beamviz import visualize

from . import egsinp
from .utils import XCITE_DIR


def reflect_x(points):
    reflected = []
    for x, y in points:
        reflected.append((-x, y))
    return reflected


def reflect_y(points):
    reflected = []
    for x, y in points:
        reflected.append((x, -y))
    return reflected


def find_x(x0, z0, x1, z1, z2):
    m = (z1 - z0) / ((x1 - x0) or 0.000001)
    x2 = (z2 - z0) / m + x0
    return x2


def translate(points, dx=0, dy=0):
    translated = []
    for x, y in points:
        translated.append((x + dx, y + dy))
    return translated


def make_target_distribution(conf, X):
    if conf['target']['distribution']:
        return [0.0] * len(X)
    elif conf['target']['distribution'] == 'left-right':
        assert len(X) % 2 == 0
        radius = conf['lesion']['diameter'] / 2
        midpoint = radius / 2
        return [-midpoint] * len(X) // 2 + [midpoint] * len(X) // 2
    elif conf['target']['distribution'] == 'polynomial':
        collimator_radius = conf['collimator']['diameter'] / 2
        lesion_radius = conf['lesion']['diameter'] / 2
        normalized = np.array(X) * collimator_radius / lesion_radius
        return list(np.polyval(conf['target']['coefficients'], normalized))


def make_anode_points(conf):
    x_radius = conf['holes']['width'] / 2
    if 'height' in conf['holes']:
        y_radius = conf['holes']['height'] / 2
    else:
        y_radius = x_radius * 2 / np.sqrt(3)
    return [
        (-x_radius, y_radius / 2),
        (-x_radius, -y_radius / 2),
        (0, -y_radius),
        (x_radius, -y_radius / 2),
        (x_radius, ),
        (0, y_radius)
    ]


def make_regions(conf):
    points = make_anode_points(conf)
    # ok now we need to 

    # just construct one row
    # start at 
    x = 0


def make_blocks_new(conf):
    block_length = conf['collimator']['length'] / conf['beamnrc']['blocks']


def make_blocks(conf):
    block_length = conf['collimator']['length'] / conf['beamnrc']['blocks']
    lesion_radius = conf['lesion']['diameter'] / 2
    target_z = conf['collimator']['length'] + conf['lesion']['distance']
    assert conf['septa']['x'] == conf['septa']['y']
    septa = conf['septa']['x']
    dy = np.sqrt(3) / 2 * lesion_radius
    dx = lesion_radius / 2
    phantom_points = [
        (-lesion_radius, 0),
        (-dx, -dy),
        (dx, -dy),
        (lesion_radius, 0),
        (dx, dy),
        (-dx, dy)
    ]

    target_points = [(0, 0) for x, y in phantom_points]

    # point them linearly at single points along, so there is no focal point?
    # ok now we're going to translate them
    phantom_regions = []  # [(phantom_points, target_points)]
    width_remaining = conf['collimator']['diameter'] / 2
    dx = lesion_radius * 2 + septa
    dy = lesion_radius * 2 + septa
    i = 0
    while width_remaining >= dx:
        width_remaining -= dx
        for j in range(conf['collimator']['rows'] // 2 + 1):
            _dx = i * dx
            _dy = j * dy
            if j % 2 == 1:
                _dx = _dx - lesion_radius
            points = translate(phantom_points, dx=_dx, dy=_dy)
            phantom_regions.append((points, target_points))
        i += 1
    for phantom_points, target_points in phantom_regions[:]:
        phantom_regions.insert(0, (reflect_y(phantom_points), reflect_y(target_points)))

    for phantom_points, target_points in phantom_regions[:]:
        phantom_regions.insert(0, (reflect_x(phantom_points), reflect_x(target_points)))

    blocks = []
    for i in range(conf['beamnrc']['blocks']):
        regions = []
        current_z = i * block_length
        for phantom_points, target_points in phantom_regions:
            region = []
            # need to translate by the center of this region
            # so where is the center?
            for (x, y), (target_x, target_y) in zip(phantom_points, target_points):
                x_ = find_x(target_x, target_z, x, conf['collimator']['length'], current_z + block_length)
                y_ = find_x(target_y, target_z, y, conf['collimator']['length'], current_z + block_length)
                region.append((x_, y_))
            regions.append(region)
        block = {
            'zmin': current_z,
            'zmax': current_z + block_length,
            'regions': regions
        }
        blocks.append(block)
    return blocks


def build_collimator(template, config):
    collimator = copy.deepcopy(template)
    blocks = make_blocks(config)
    for i, block in enumerate(blocks):
        cm = {
            'type': 'BLOCK',
            'identifier': 'BLCK{}'.format(i),
            'rmax_cm': config['beamnrc']['rmax'],
            'title': 'BLCK{}'.format(i),
            'zmin': block['zmin'],
            'zmax': block['zmax'],
            'zfocus': config['collimator']['length'] + config['lesion']['distance'],
            'xpmax': config['beamnrc']['rmax'],
            'ypmax': config['beamnrc']['rmax'],
            'xnmax': -config['beamnrc']['rmax'],
            'ynmax': -config['beamnrc']['rmax'],
            'air_gap': {
                'ecut': 0.811,
                'pcut': 0.01,
                'dose_zone': 0,
                'iregion_to_bit': 0
            },
            'opening': {
                'ecut': 0.811,
                'pcut': 0.01,
                'dose_zone': 0,
                'iregion_to_bit': 0,
                'medium': 'Air_516kVb'
            },
            'block': {
                'ecut': 0.521,
                'pcut': 0.01,
                'dose_zone': 0,
                'iregion_to_bit': 0,
                'medium': 'PB516'
            },
            'regions': []
        }
        for region in block['regions']:
            cm['regions'].append({
                'points': [{'x': x, 'y': y} for x, y in region]
            })
        collimator['cms'].append(cm)
    return collimator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', default='collimator.toml')
    parser.add_argument('output', nargs='?')
    parser.add_argument('--template')
    args = parser.parse_args()
    if not args.output:
        args.output = os.path.splitext(args.config)[0] + '.generated.egsinp'
    if not args.template:
        args.template = os.path.join(XCITE_DIR, 'templates/template.egsinp')
    return args


def load_template(path):
    with open(path) as f:
        return egsinp.parse_egsinp(f.read())


def load_config(path):
    with open(path) as f:
        return toml.load(f)


def save_collimator(collimator, path):
    with open(path, 'w') as f:
        f.write(egsinp.unparse_egsinp(collimator))


def save_config(config, path):
    config['revision'] = get_revision()
    lines = []
    for line in toml.dumps(config).splitlines():
        if line.startswith('['):
            lines.append('')
        lines.append(line)
    with open(path, 'w') as f:
        f.write('\n'.join(lines))


def get_revision():
    return subprocess.run(['git', 'rev-parse', 'HEAD'], stdout=subprocess.PIPE).stdout.decode('utf-8').strip()


def main():
    args = parse_args()
    config = load_config(args.config)
    template = load_template(args.template)
    collimator = build_collimator(template, config)
    save_collimator(collimator, args.output)
    save_config(config, os.path.splitext(args.output)[0] + '.toml')
    visualize.render(args.output, config['lesion']['diameter'])


if __name__ == '__main__':
    sys.exit(main())

    """
    jout = open(args.output.replace('.egsinp', '.json'), 'w')
    data = vars(args)
    data['revision'] = get_revision()
    json.dump(data, jout, sort_keys=True, indent=2)
    """
