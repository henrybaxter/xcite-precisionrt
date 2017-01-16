import time
import math
import sys
import os
import pytoml as toml
import subprocess
import argparse
import logging
import copy
import itertools

import numpy as np
from beamviz import visualize
from shapely.geometry import Polygon

from . import egsinp
from .utils import XCITE_DIR

logger = logging.getLogger(__name__)


def map_to_circle(ingress, ingress_center, target_center, radius):
    translated = ingress - ingress_center
    norms = np.sqrt(np.sum(np.square(translated), axis=1))[:, np.newaxis]
    normalized = translated / norms
    result = normalized * radius + target_center
    return result


def clip_y(vectors, min_value, max_value):
    return np.array([
        vectors[:, 0],
        np.clip(vectors[:, 1], min_value, max_value)
    ]).T


def make_egress(conf, ingress, ingress_center):
    target_center = make_target_center(conf, ingress_center)
    target = make_target(conf, ingress, ingress_center, target_center)
    zfocus = conf['length'] + conf['lesion-distance']
    egress = ingress + (target - ingress) / zfocus * conf['length']
    return egress


def make_target_center(conf, ingress_center):
    x = ingress_center[0]
    collimator_radius = conf['diameter'] / 2
    lesion_radius = conf['lesion-diameter'] / 2
    normalized = x / collimator_radius * lesion_radius
    if conf['target-distribution'] == 'center':
        coefficients = [0]
    elif conf['target-distribution'] == 'left-right':
        lesion_radius = conf['lesion']['diameter'] / 2
        midpoint = lesion_radius / 2
        coefficients = [math.copysign(midpoint, x)]
    elif conf['target-distribution'] == 'polynomial':
        coefficients = conf['target-coefficients']
    else:
        raise ValueError('Unknown distribution {}'.format(conf['target-distribution']))
    return np.array([np.polyval(coefficients, normalized), 0])


def make_target(conf, ingress, ingress_center, target_center):
    if conf['target-shape'] == 'point':
        return map_to_circle(ingress, ingress_center, target_center, radius=0)
    elif conf['target-shape'] == 'line':
        radius = conf['lesion-diameter'] / 2
        return clip_y(map_to_circle(ingress, ingress_center, target_center, radius), 0, 0)
    elif conf['target-shape'] == 'circle':
        return map_to_circle(ingress, ingress_center, target_center, conf['lesion-diameter'] / 2)
    else:
        raise ValueError('Unknown target shape {}'.format(conf['target-shape']))


def interpolate(ingress, egress, z, zmax):
    if z == 0:
        return ingress
    return ingress + (egress - ingress) / zmax * z


def calculate_dy(conf, ingress, center, egress, dx, iterations=1000):
    septa = conf['septa-y']
    precision = conf['precision']
    height = np.amax(ingress, axis=0)[1] - np.amin(ingress, axis=0)[1]
    for i in range(iterations):
        dy = np.array([0, height / 2 + septa + i * precision])
        if Polygon(egress).distance(Polygon(make_egress(conf, ingress + dy + dx / 2, center + dy + dx / 2))) > septa:
            return dy
    raise ValueError('calculate_dx failed after {} iterations'.format(iterations))


def calculate_dx(conf, ingress, center, egress, iterations=1000):
    septa = conf['septa-x']
    precision = conf['precision']
    width = np.amax(ingress, axis=0)[0] - np.amin(ingress, axis=0)[0]
    for i in range(iterations):
        dx = np.array([width + septa + i * precision, 0])
        next_egress = make_egress(conf, ingress + dx, center + dx)
        if Polygon(egress).distance(Polygon(next_egress)) > septa:
            return dx
    raise ValueError('calculate_dy failed after {} iterations'.format(iterations))


def make_blocks(conf):
    if conf['rows'] % 2 == 0:
        raise ValueError('Only an odd number of rows is supported')
    conf['septa-x'] = conf.get('septa-x', conf.get('septa'))
    conf['septa-y'] = conf.get('septa-y', conf.get('septa'))
    if conf['septa-x'] < 0 or conf['septa-y'] < 0:
        raise ValueError('Only non-negative septa values are supported')
    if 'hole-width' in conf and 'hole-height' in conf:
        x_radius = conf['hole-width'] / 2
        y_radius = conf['hole-height'] / 2
    elif 'hole-width' in conf:
        x_radius = conf['hole-width'] / 2
        y_radius = x_radius * 2 / np.sqrt(3)
    elif 'hole-height' in conf:
        y_radius = conf['hole-height'] / 2
        x_radius = y_radius * np.sqrt(3) / 2
    else:
        raise KeyError('hole-width or hole-height or both required')
    ingress = np.array([
        # the order matters for ease of constructing the endcaps
        (-x_radius, -y_radius / 2),
        (0, -y_radius),
        (x_radius, -y_radius / 2),
        (x_radius, y_radius / 2),
        (0, y_radius),
        (-x_radius, y_radius / 2)
    ])
    center = np.array([0.0, 0.0])
    egress = make_egress(conf, ingress, center)
    dx = calculate_dx(conf, ingress, center, egress)
    rows = [(center, ingress, egress)]
    max_radius = conf['diameter'] / 2
    for i in itertools.count(1):
        xy = i * dx
        if np.amax(ingress + xy, axis=0)[0] > max_radius:
            break
        rows.append((center + xy, ingress + xy, make_egress(conf, ingress + xy, center + xy)))

    dy = calculate_dy(conf, ingress, center, egress, dx)

    # reflect everything (around x)
    for center, ingress, egress in rows[1:]:
        xy, ingress, egress = np.copy(center), np.copy(ingress), np.copy(egress)
        xy[0] *= -1
        ingress[:, 0] *= -1
        egress[:, 0] *= -1
        rows.insert(0, (xy, ingress, egress))

    columns = len(rows)

    # now add the rows
    for center, ingress, egress in rows[:]:
        for i in range(1, conf['rows'] // 2 + 1):
            xy = i * dy + (i % 2) * dx / 2
            rows.append((center + xy, ingress + xy, make_egress(conf, ingress + xy, center + xy)))
        i += 1
        xy = i * dy + (i % 2) * dx / 2
        # here we want to chop the ingress a bit.
        # take only the first 3 points
        if conf['rowcaps']:
            rows.append((center + xy, ingress[:3] + xy, make_egress(conf, ingress[:3] + xy, center[:3] + xy)))

    # finally reflect around y, BUT skip the first row
    for xy, ingress, egress in rows[columns:]:
        xy, ingress, egress = np.copy(xy), np.copy(ingress), np.copy(egress)
        xy[1] *= -1
        ingress[:, 1] *= -1
        egress[:, 1] *= -1
        rows.append((xy, ingress, egress))

    pairs = []
    for xy, ingress, egress in rows:
        pairs.append((ingress, egress))

    # now let's build some rows.
    # this is a bit more complicated isn't it.
    # what they focus on is a bit different

    block_length = conf['length'] / conf['blocks']
    blocks = []
    for i in range(conf['blocks']):
        z = i * block_length
        regions = []
        for ingress, egress in pairs:
            # interpolate between them
            # ingress is at z = 0
            # egress is at z = conf['length']
            # need to find the point between
            # now since the
            regions.append(interpolate(ingress, egress, i * block_length, conf['length']))
            # here we need to interpolate between them
        blocks.append({
            'zmin': z,
            'zmax': z + block_length,
            'regions': regions
        })
    return blocks


def make_collimator(template, config):
    with open('collimator.defaults.toml') as f:
        defaults = toml.load(f).copy()
        defaults.update(config)
        config.clear()
        config.update(defaults)
    collimator = copy.deepcopy(template)
    blocks = make_blocks(config)
    for i, block in enumerate(blocks):
        cm = {
            'type': 'BLOCK',
            'identifier': 'BLCK{}'.format(i),
            'rmax_cm': config['rmax'],
            'title': 'BLCK{}'.format(i),
            'zmin': block['zmin'],
            'zmax': block['zmax'],
            'zfocus': config['length'] + config['lesion-distance'],
            'xpmax': config['rmax'],
            'ypmax': config['rmax'],
            'xnmax': -config['rmax'],
            'ynmax': -config['rmax'],
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
    return subprocess.run(
        ['git', 'rev-parse', 'HEAD'],
        stdout=subprocess.PIPE
    ).stdout.decode('utf-8').strip()


def main():
    start = time.time()
    logging.basicConfig(level=logging.DEBUG)
    args = parse_args()
    config = load_config(args.config)
    template = load_template(args.template)
    collimator = make_collimator(template, config)
    save_collimator(collimator, args.output)
    save_config(config, os.path.splitext(args.output)[0] + '.toml')
    visualize.render(args.output, config['lesion-diameter'])
    elapsed = time.time() - start
    logger.info('Took {:.2f} seconds'.format(elapsed))


if __name__ == '__main__':
    sys.exit(main())
