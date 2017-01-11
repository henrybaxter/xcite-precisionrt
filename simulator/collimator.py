import math
import sys
import os
import toml
import subprocess
import argparse
import logging
import copy
from operator import itemgetter

import numpy as np
from beamviz import visualize

from . import egsinp
from .utils import XCITE_DIR

logger = logging.getLogger(__name__)

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


def make_target_x(conf, x):
    logger.info('Making target with ingress_x {}'.format(x))
    collimator_radius = conf['collimator']['diameter'] / 2
    lesion_radius = conf['lesion']['diameter'] / 2
    normalized = x / collimator_radius * lesion_radius
    logger.info('Normalized is {}'.format(normalized))
    if conf['target']['distribution'] == 'center':
        coefficients = [0]
    elif conf['target']['distribution'] == 'left-right':
        lesion_radius = conf['lesion']['diameter'] / 2
        midpoint = lesion_radius / 2
        coefficients = [math.copysign(midpoint, x)]
    elif conf['target']['distribution'] == 'polynomial':
        coefficients = conf['target']['coefficients']
    else:
        raise ValueError('Unknown distribution {}'.format(conf['target']['distribution']))
    logger.info('coefficients = {}, normalized = {}'.format(coefficients, normalized))
    result = np.polyval(coefficients, normalized)
    logger.info('result = {}'.format(result))
    return result


def make_target(conf, ingress, center):
    logger.info('Making target of ingress {} and center {}'.format(ingress, center))
    centers = np.repeat(center, ingress.shape[0]).reshape(ingress.shape)
    if conf['target']['mapto'] == 'point':
        return centers
    elif conf['target']['mapto'] == 'rectangle':
        # rectangle is relative to center
        size = np.array([conf['target']['width'], conf['target']['height']])
        print('size', size)
        result = centers + np.copysign(size / 2, ingress - centers)
        print('result', result)
        return result
    elif conf['target']['mapto'] == 'circle':
        norms = np.sqrt(np.sum(np.square(ingress), axis=1))
        return (ingress - center) / norms[:, np.newaxis] * conf['target']['radius']
    elif conf['target']['mapto'] == 'chords':
        # here we map to a circle, but clamp the x
        # so the chord is of width conf['target']['width']
        # and the height of the chord is dictated by the radius
        norms = np.sqrt(np.sum(np.square(ingress), axis=1))
        circle = (ingress - center) / norms[:, np.newaxis] * conf['target']['radius']
        half_width = conf['target']['width'] / 2
        return np.array([
            np.clip(circle[:, 0], center[0] - half_width, center[0] + half_width),
            circle[:, 1]
        ]).T


def make_egress(ingress, target, z, zmax):
    z0 = 0
    z1 = z
    z2 = zmax
    return ingress + (target - ingress) / (z2 - z0) * z1


def make_ingress(egress, target, z, zmax):
    z0 = 0
    z1 = z
    z2 = zmax
    return target + (egress - target) / (z2 - z1) * (z2 - z0)


def make_blocks(conf):
    logger.debug('Making blocks')
    x_radius = conf['holes']['width'] / 2
    if 'height' in conf['holes']:
        y_radius = conf['holes']['height'] / 2
    else:
        y_radius = x_radius * 2 / np.sqrt(3)
    logger.debug('Radii are ({}, {})'.format(x_radius, y_radius))
    start = np.array([
        (-x_radius, y_radius / 2),
        (-x_radius, -y_radius / 2),
        (0, -y_radius),
        (x_radius, -y_radius / 2),
        (x_radius, y_radius / 2),
        (0, y_radius)
    ])
    logger.debug('Initial points are {}'.format(start))
    # generate one side!
    # center = np.array([0, 0])
    ingress_center = np.array([0.0, 0.0])
    zfocus = conf['collimator']['length'] + conf['lesion']['distance']
    ingress = start[:]
    regions = []
    i = 0

    # while the maximum x value of the ingress is less than the diameter, we're good
    while True:
        max_x = np.amax(ingress, axis=0)[0]
        print('max_x =', max_x)
        collimator_radius = conf['collimator']['diameter'] / 2
        print('collimator radius =', collimator_radius)
        if max_x > collimator_radius:
            break
        i += 1
        print('Iteration {}'.format(i))
        target_x = make_target_x(conf, ingress_center[0])
        print('target_x = {}'.format(target_x))
        target = make_target(conf, ingress, np.array([target_x, 0]))
        print('target = {}'.format(target))
        egress = make_egress(ingress, target, conf['collimator']['length'], zfocus)
        print('egress = {}'.format(egress))
        print('added to regions')
        regions.append(egress)

        min_ingress_x = np.amax(ingress, axis=0)[0] + conf['septa']['x']
        min_egress_x = np.amax(egress, axis=0)[0] + conf['septa']['x']
        print('min_ingress_x {}'.format(min_ingress_x))
        print('min_egress_x {}'.format(min_egress_x))

        def width(points):
            return np.amax(points, axis=0)[0] - np.amin(points, axis=0)[0]

        ingress_dx = np.array([width(ingress) + conf['septa']['x'], 0])
        egress_dx = np.array([width(egress) + conf['septa']['x'], 0])
        candidate_ingress = ingress + ingress_dx
        candidate_egress = egress + egress_dx
        print('candidate_ingress = {}'.format(candidate_ingress))
        print('candidate_egress = {}'.format(candidate_egress))
        resulting_egress = make_egress(candidate_ingress, target, conf['collimator']['length'], zfocus)
        resulting_ingress = make_ingress(candidate_egress, target, conf['collimator']['length'], zfocus)
        print('resulting_egress = {}'.format(resulting_egress))
        print('resulting_ingress = {}'.format(resulting_ingress))
        if np.amin(resulting_egress, axis=0)[0] >= min_egress_x:
            print('Shuffling along ingress')
            # shuffling along ingress worked, move center that far
            ingress_center += ingress_dx
            ingress = candidate_ingress
            egress = resulting_egress
        elif np.amin(resulting_ingress, axis=0)[0] >= min_ingress_x:
            # TODO choose a 'closest' egress point, shift by the septa, then
            # backtrack up to the corresponding ingress point. 
            # pick the furthest point, find it
            print('Shuffling along egress')
            ingress_dx = np.amin(resulting_ingress - ingress, axis=0)[0]
            ingress = resulting_ingress
            egress = candidate_egress
        else:
            raise ValueError('Impossible to construct collimator without overlap')
        print('ingress {}'.format(ingress))
        print('egress {}'.format(egress))


    # once we have the start and endpoint everywhere, we just interpolate the shit out of that
    # but for now, screw it
    # block_length = conf['collimator']['length'] / conf['beamnrc']['blocks']
    # for i in range(conf['beamnrc']['blocks']):
        # z = i * block_length
    # ok, so now we have the region points!
    block = {
        'zmin': 0,
        'zmax': conf['collimator']['length'],
        'regions': regions
    }
    return [block]


def _make_blocks(conf):
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
    logging.basicConfig(level=logging.DEBUG)
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
