import datetime
import time
import shutil
import platform
import argparse
import sys
import os
import math
import hashlib
import json
import functools
import logging
from collections import OrderedDict
import asyncio

import numpy as np

import report
import egsinp
import grace
import collimator_analyzer
import dose_contours
import py3ddose
import simulate

import build
from utils import run_command

logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--plot-config', action='append', default=['grace.json'])

    # common
    parser.add_argument('name')
    parser.add_argument('--egsinp-template', default='template.egsinp')
    parser.add_argument('--pegs4', default='allkV')
    parser.add_argument('--rmax', type=float, default=50.0,
                        help='Max extent of all component modules')
    parser.add_argument('--overwrite', action='store_true')

    # source arguments
    parser.add_argument('--beam-width', type=float, default=1,
                        help='Beam width (y) in cm')
    parser.add_argument('--beam-height', type=float, default=0.5,
                        help='Beam height (z) in cm')
    parser.add_argument('--beam-distance', type=float, default=50.0,
                        help='Beam axis of rotation to target in cm')
    parser.add_argument('--target-length', type=float, default=75.0,
                        help='Length of target in cm')
    parser.add_argument('--target-angle', default=45.0, type=float,
                        help='Angle of reflection target')
    parser.add_argument('--beam-gap', type=float, default=0.0,
                        help='Gap between incident beams')
    parser.add_argument('--histories', type=int, default=int(1e9),
                        help='Divided among source beamlets')

    # collimator
    #   generated
    parser.add_argument('--target-distance', type=float, default=None,
                        help='Distance from end of collimator to isocenter')
    #   given
    parser.add_argument('--collimator', required=True,
                        help='Input egsinp path or use stamped values')

    # dose
    parser.add_argument('--phantom', default='cylindricalp.egsphant',
                        help='.egsphant file')
    parser.add_argument('--target-z', type=float, default=-10,
                        help='Isocenter z coordinate')
    parser.add_argument('--target-y', type=float, default=20,
                        help='Isocenter y coordinate')
    parser.add_argument('--target-x', type=float, default=0,
                        help='Isocenter x coordinate')
    parser.add_argument('--target-size', type=float, default=1.0,
                        help='Target size')
    parser.add_argument('--dose-egsinp', default='dose_template.egsinp',
                        help='.egsinp for dosxyznrc')
    parser.add_argument('--arc-dose-egsinp', default='arc_dose_template.egsinp',
                        help='.egsinp for arced dosxyznrc')
    parser.add_argument('--dose-recycle', type=int, default=9,
                        help='Use particles n + 1 times')
    parser.add_argument('--dose-photon-splitting', type=int, default=20,
                        help='n_split in dose egsinp')

    args = parser.parse_args()
    args.output_dir = args.name.replace(' ', '-')
    if os.path.exists(args.output_dir):
        logger.warning('{} already exists'.format(args.output_dir))
    for subfolder in ['dose/stationary', 'dose/arc']:
        os.makedirs(os.path.join(args.output_dir, subfolder), exist_ok=True)

    args.egs_home = os.path.abspath(os.path.join(
        os.environ['HEN_HOUSE'], '../egs_home/'))
    logger.info('egs_home is {}'.format(args.egs_home))

    return args


def sample_combine(beamlets, desired=10000000):
    logger.info('Sampling and combining {} beamlets'.format(len(beamlets)))
    paths = [beamlet['phsp'] for beamlet in beamlets]
    particles = sum([beamlet['stats']['total_particles'] for beamlet in beamlets])
    rate = math.ceil(particles / desired)
    logger.info('Found {} particles, want {}, sample rate is {}'.format(particles, desired, rate))
    s = 'rate={}'.format(rate) + ''.join([beamlet['hash'].hexdigest() for beamlet in beamlets])
    md5 = hashlib.md5(s.encode('utf-8'))
    os.makedirs('combined', exist_ok=True)
    combined_path = 'combined/{}.egsphsp1'.format(md5.hexdigest())
    if os.path.exists(combined_path):
        logger.info('Combined beamlets file {} already exists'.format(combined_path))
        return combined_path
    logger.info('Combining {} beamlets into {}'.format(len(beamlets), combined_path))
    run_command(['beamdpr', 'sample-combine', '--rate', str(rate), '-o', combined_path] + paths)
    logger.info('Randomizing')
    run_command(['beamdpr', 'randomize', combined_path])
    return combined_path


def generate_y(target_length, spacing):
    logger.info('Generating beam positions')
    offset = spacing / 2
    y = offset
    ymax = target_length / 2
    i = 0
    result = []
    while y < ymax:
        result.append(y)
        i += 1
        y = i * spacing + offset
    # could be removed and the beams reflected instead
    # this was written before beamdpr
    for y in result[:]:
        result.insert(0, -y)
    return result


def combine_fast_doses(doses):
    logger.info('Combining doses')
    result = {}
    sz = len(doses['stationary'])
    coeffs = np.polyfit([0, sz // 2, sz - 1], [4, 1, 4], 2)
    w = np.polyval(coeffs, np.arange(0, sz))
    weights = {
        'weighted': w,
        'arc_weighted': w
    }
    doses = {
        'stationary': doses['stationary'],
        'weighted': doses['stationary'],
        'arc': doses['arc'],
        'arc_weighted': doses['arc']
    }
    for stage, beamlet_doses in doses.items():
        paths = [dose['dose'] + '.npz' for dose in beamlet_doses]
        path = os.path.join(args.output_dir, '{}.3ddose'.format(stage))
        if os.path.exists(path):
            logger.warning('Combined dose {} already exists'.format(path))
        else:
            logger.info('Combining {}'.format(stage))
            if stage in weights:
                py3ddose.weight_3ddose(paths, path, weights[stage])
            else:
                py3ddose.combine_3ddose(paths, path)
            py3ddose.read_3ddose(path)
        result[stage] = path
    return result


def combine_slow_doses(contributions):
    """Assumes contributions is a list of dictionaries, each containing entries like
    (theta, phi): { 'dose': path, 'hash': object }
    """
    center_paths = []  # beamlet contributions without arc
    arced_paths = []  # beamlet contributions with arc
    for i, beamlet_contribution in enumerate(contributions):
        beamlet_paths = []
        for (theta, phi), contribution in beamlet_contribution.items():
            if theta == 180 and phi == 0:
                center_paths.append(contribution['dose'])
            beamlet_paths.append(contribution['dose'])
        logger.info('Combining arced doses for beamlet {}'.format(i))
        arced_path = os.path.join(args.output_dir, 'dose/arc.dose{}.3ddose.npz'.format(i))
        py3ddose.combine_3ddose(beamlet_paths, arced_path)
        arced_paths.append(arced_path)
    logger.info('Combining center doses')
    py3ddose.combine_3ddose(center_paths, os.path.join(args.output_dir, 'center.3ddose.npz'))
    logger.info('Combining arced doses')
    py3ddose.combine_3ddose(arced_paths, os.path.join(args.output_dir, 'arced.3ddose.npz'))


async def configure_logging(args):
    revision = (await run_command(['git', 'rev-parse', '--short', 'HEAD'])).strip()
    formatter = logging.Formatter('%(levelname)s {name} {revision} %(asctime)s %(message)s'.format(
        revision=revision, name=args.name), '%H:%M:%S')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    today = datetime.date.today()
    filename = 'simulation.{}.{}.log'.format(args.name.replace(' ', ''), str(today))
    file_handler = logging.FileHandler(filename)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler)
    logging.getLogger().addHandler(file_handler)
    logging.getLogger().setLevel(logging.DEBUG)


def log_args(args):
    printable_args = [(k, v) for k, v in sorted(args.__dict__.items())]
    pretty_args = '\n'.join(['\t{}: {}'.format(k, v) for k, v in printable_args])
    logger.info('Arguments: \n{}'.format(pretty_args))


async def main():
    args = parse_args()
    await configure_logging(args)
    log_args(args)
    y_values = generate_y(args.target_length, args.beam_width + args.beam_gap)
    histories = args.histories // len(y_values)
    stages = ['source', 'filter', 'collimator', 'stationary_dose', 'arc_dose']
    templates = {stage: template for stage, template in zip(stages, await asyncio.gather(*[
        build.build_source(args, histories),
        build.build_filter(args),
        build.build_collimator(args)
    ]))}
    templates['stationary_dose'] = open(args.dose_egsinp).read()
    templates['arc_dose'] = open(args.arc_dose_egsinp).read()
    simulations = await asyncio.gather(*[
        simulate.simulate(args, templates, i, y) for i, y in enumerate(y_values)
    ])
    asyncio.gather(*[
        # combine source, filter, collimator
        # combine stationary and arc doses
        # guess weights/apply weights and combine stationary weighted and arc weighted
        # generate contour plots
        # generate grace plots
    ])
    # post process some stats
    # generate tex file
    # generate report


    """
    result = {
        'source': {
            'egsinp': full path,
            'in_phsp': full path,
            'out_phsp': full path,
            'in_stats': stats of in_phsp,
            'out_stats': stats of out_phsp,
            'egslst': full path to egslst output of simulation,
        },
        'filter'...,
        'collimator'...,
        'stationary_dose': {
            'egsinp'...,
            'in_phsp',
            'in_stats'...,
            'out_3ddose',
            'egslst'...
        }
        'arced_dose': {
            'in_phsp':
            'in_stats',
            'doses': [{ # these are unordered
                'index':...
                'egsinp',
                '3ddose',
                'egslst'
            }]
        }
    }

    combined_result = {
        'source': {
            'phsp',
            'stats'
        },
        'filter': {
        ...
        }
    }
    """
    """"
    # now what about this sampling and combining stuff
    # we do that at the end, based on this datastructure?
    # 
    # now, we need to eventually order these, so...
    phsp['source'] = sample_combine(beamlets['source'])
    shutil.copy(phsp['source'], os.path.join(args.output_dir, 'sampled_source.egsphsp1'))

    _filter = build_filter(args)
    beamlets['filter'] = beamlet_stats(filter_source(beamlets['source'], _filter, args))
    phsp['filter'] = sample_combine(beamlets['filter'])
    shutil.copy(phsp['filter'], os.path.join(args.output_dir, 'sampled_filter.egsphsp1'))

    collimator = build_collimator(args)
    beamlets['collimator'] = beamlet_stats(collimate(beamlets['filter'], collimator, args))
    phsp['collimator'] = sample_combine(beamlets['collimator'], desired=100000000)
    shutil.copy(phsp['collimator'], os.path.join(args.output_dir, 'sampled_collimator.egsphsp1'))

    # dose_contributions = dose(beamlets['collimator'], args)
    # combine_doses(dose_contributions)
    doses = fast_dose(beamlets['collimator'], args)  # stationary, arc
    combined_doses = combine_fast_doses(doses)
    target = py3ddose.Target(
        np.array([args.target_z, args.target_y, args.target_x]),
        args.target_size)
    contours = {}
    conformity = {}
    target_to_skin = {}
    for slug, path in combined_doses.items():
        contours[slug] = dose_contours.plot(args.phantom, path, target, args.output_dir, '{}_dose'.format(slug))
        dose = py3ddose.read_3ddose(path)
        conformity[slug] = py3ddose.paddick(dose, target)
        target_to_skin[slug] = py3ddose.target_to_skin(dose, target)

    # we take the plane
    contour_plots = OrderedDict()
    for stage in ['stationary', 'weighted', 'arc', 'arc_weighted']:
        for contour in contours[stage]:
            contour_plots.setdefault(contour['plane'], []).append(contour)

    photons = {}
    for stage in ['source', 'filter', 'collimator']:
        photons[stage] = sum([beamlet['stats']['total_photons'] for beamlet in beamlets[stage]])

    data = {
        '_filter': _filter,
        'collimator': collimator,
        'collimator_stats': collimator_analyzer.analyze(collimator),
        'beamlets': beamlets,
        'phsp': phsp,
        'grace_plots': grace.make_plots(args.output_dir, phsp, args.plot_config),
        'contour_plots': contour_plots,
        'skin_distance': args.target_distance - abs(args.target_z),
        'ci': conformity,
        'ts': target_to_skin,
        'electrons': histories,
        'photons': photons
    }
    report.generate(data, args)
    """
    logger.info('Output in {}'.format(args.output_dir))


if __name__ == '__main__':
    start = time.time()
    loop = asyncio.get_event_loop()
    #import warnings
    #loop.set_debug(True)
    #loop.slow_callback_duration = 0.001
    #warnings.simplefilter('always', ResourceWarning)
    loop.run_until_complete(main())
    loop.close()
    logger.info('Finished in {:.2f} seconds'.format(time.time() - start))
