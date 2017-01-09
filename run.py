import sys
import datetime
import time
import argparse
import os
import math
import hashlib
import logging
import asyncio
from collections import OrderedDict

import numpy as np

import collimator_analyzer
import py3ddose
import grace
import simulate
import dose_contours
import build
from utils import run_command, copy, read_3ddose
import report

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
    parser.add_argument('--reflect', action='store_true')

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
        logger.error('{} already exists'.format(args.output_dir))
        sys.exit(1)
    for subfolder in ['dose/stationary', 'dose/arc']:
        os.makedirs(os.path.join(args.output_dir, subfolder), exist_ok=True)

    args.egs_home = os.path.abspath(os.path.join(
        os.environ['HEN_HOUSE'], '../egs_home/'))
    logger.info('egs_home is {}'.format(args.egs_home))

    return args


async def sample_combine(beamlets, reflect, desired=int(1e7)):
    logger.info('Sampling and combining {} beamlets'.format(len(beamlets)))
    paths = [beamlet['phsp'] for beamlet in beamlets]
    particles = sum([beamlet['stats']['total_particles'] for beamlet in beamlets])
    if reflect:
        desired // 2
    rate = math.ceil(particles / desired)
    logger.info('Found {} particles, want {}, sample rate is {}'.format(particles, desired, rate))
    s = 'rate={}&reflecty={}'.format(rate, reflect) + ''.join([beamlet['hash'].hexdigest() for beamlet in beamlets])
    md5 = hashlib.md5(s.encode('utf-8'))
    os.makedirs('combined', exist_ok=True)
    temp_path = 'combined/{}.egsphsp1'.format(md5.hexdigest())
    combined_path = 'combined/{}.egsphsp'.format(md5.hexdigest())
    if not os.path.exists(combined_path):
        logger.info('Combining {} beamlets into {}'.format(len(beamlets), temp_path))
        await run_command(['beamdpr', 'sample-combine', '--rate', str(rate), '-o', temp_path] + paths)
        if reflect:
            original_path = temp_path.replace('.egsphsp1', '.original.egsphsp1')
            reflected_path = temp_path.replace('.egsphsp1', '.reflected.egsphsp1')
            os.rename(temp_path, original_path)
            await run_command(['beamdpr', 'reflect', '-y', '1', original_path, reflected_path])
            await run_command(['beamdpr', 'combine', original_path, reflected_path, '-o', temp_path])
        logger.info('Randomizing {}'.format(temp_path))
        await run_command(['beamdpr', 'randomize', temp_path])
        os.rename(temp_path, combined_path)
    return combined_path


def generate_y(target_length, spacing, reflect):
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
    if not reflect:
        # need to reflect y values if not using reflection optimization
        for y in result[:]:
            result.insert(0, -y)
    return result


async def combine_doses(args, doses):
    """
    what is the purpose here? how do we weight the arcs?
    the problem is the arc dose seems different
    but we can just flatten it out

    how do we weight the arc dose?
    """
    logger.info('Combining doses')
    result = {}
    sz = len(doses['stationary'])
    coeffs = np.polyfit([0, sz // 2, sz - 1], [4, 1, 4], 2)
    w = np.polyval(coeffs, np.arange(0, sz))
    weights = {
        'stationary_weighted': w,
    }
    doses = {
        'stationary': doses['stationary'],
        'stationary_weighted': doses['stationary'],
        'arc': doses['arc'],
        'arc_weighted': doses['arc']
    }
    for stage, beamlet_doses in doses.items():
        if 'arc' in stage:  # flatten
            beamlet_doses = [d for ds in beamlet_doses for d in ds]
        paths = [dose['npz'] for dose in beamlet_doses]
        path = os.path.join(args.output_dir, '{}.3ddose'.format(stage))
        if os.path.exists(path):
            logger.warning('Combined dose {} already exists'.format(path))
        else:
            logger.info('Combining {}'.format(stage))
            if stage in weights:
                py3ddose.weight_3ddose(paths, path, weights[stage])
            else:
                py3ddose.combine_3ddose(paths, path)
            await read_3ddose(path)
        result[stage] = path
    return result


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
    y_values = generate_y(args.target_length, args.beam_width + args.beam_gap, args.reflect)
    histories = args.histories // len(y_values)
    if args.reflect:
        histories //= 2
    stages = ['source', 'filter', 'collimator', 'dose']
    templates = {stage: template for stage, template in zip(stages, await asyncio.gather(*[
        build.build_source(args, histories),
        build.build_filter(args),
        build.build_collimator(args)
    ]))}
    with open(args.dose_egsinp) as f:
        templates['stationary_dose'] = f.read()
    with open(args.arc_dose_egsinp) as f:
        templates['arc_dose'] = f.read()
    simulations = []
    for i, y in enumerate(y_values):
        if args.reflect:
            index = (len(y_values) - i - 1, i + len(y_values))
        else:
            index = i
        simulations.append(simulate.simulate(args, templates, index, y))
    simulations = await asyncio.gather(*simulations)
    logger.info('All simulations finished, combining')
    beamlets = {
        'source': [sim['source'] for sim in simulations],
        'filter': [sim['filter'] for sim in simulations],
        'collimator': [sim['collimator'] for sim in simulations],
    }
    doses = {}
    if args.reflect:
        # dose calculations were put into tuples
        for key in ['stationary', 'arc']:
            full = []
            for simulation in simulations:
                from_reflected, from_original = simulation['dose']
                full.insert(0, from_reflected[key])
                full.append(from_original[key])
                doses[key] = full
    else:
        for key in ['stationary', 'arc']:
            full = []
            for simulation in simulations:
                full.append(simulation['dose'][key])
            doses[key] = full

    combined = {stage: result for stage, result in zip(stages, await asyncio.gather(*[
        sample_combine(beamlets['source'], args.reflect),
        sample_combine(beamlets['filter'], args.reflect),
        sample_combine(beamlets['collimator'], args.reflect),
        combine_doses(args, doses)
    ]))}

    logger.info('Saving combined phase space files')
    await asyncio.gather(*[
        copy(combined[key], os.path.join(args.output_dir, 'sampled_{}.egsphsp'.format(key)))
        for key in ['source', 'filter', 'collimator']
    ])

    # plots
    plot_futures = [
        grace.make_plots(args.output_dir, combined, args.plot_config),
    ]
    target = py3ddose.Target(
        np.array([args.target_z, args.target_y, args.target_x]),
        args.target_size)
    for slug, path in combined['dose'].items():
        plot_futures.append(dose_contours.plot(args.phantom, path, target, args.output_dir, slug))
    grace_plots, *contours = await asyncio.gather(*plot_futures)
    contour_plots = OrderedDict()
    for stage in ['stationary', 'stationary_weighted', 'arc', 'arc_weighted']:
        for contour in [c for cs in contours for c in cs]:
            if contour['output_slug'] == stage:
                contour_plots.setdefault(contour['plane'], []).append(contour)
    logger.info('Generating conformity and target to skin ratios')
    conformity = {}
    target_to_skin = {}
    for slug, path in combined['dose'].items():
        dose = py3ddose.read_3ddose(path)
        conformity[slug] = py3ddose.paddick(dose, target)
        target_to_skin[slug] = py3ddose.target_to_skin(dose, target)

    logger.info('Getting photons')
    photons = {}
    for stage in ['source', 'filter', 'collimator']:
        photons[stage] = sum([sim[stage]['stats']['total_photons'] for sim in simulations])
    data = {
        '_filter': templates['filter'],
        'collimator': templates['collimator'],
        'collimator_stats': collimator_analyzer.analyze(templates['collimator']),
        'grace_plots': grace_plots,
        'contour_plots': contour_plots,
        'skin_distance': args.target_distance - abs(args.target_z),
        'ci': conformity,
        'ts': target_to_skin,
        'electrons': histories * len(y_values),
        'photons': photons
    }
    report.generate(data, args)
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
