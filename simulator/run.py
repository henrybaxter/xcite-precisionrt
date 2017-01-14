import io
import time
import argparse
import os
import math
import hashlib
import pytoml as toml
import logging
import asyncio
from collections import OrderedDict

import numpy as np
import boto3
from botocore.exceptions import ClientError as BotoClientError

from . import collimator_analyzer
from . import py3ddose
from . import grace
from . import simulate
from . import dose_contours
from . import build
from .utils import run_command, read_3ddose, force_symlink
from . import report

logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


async def main():
    configure_logging()
    args = parse_args()
    local = read_local()
    simulations = read_simulations(args)
    for sim in simulations:
        sim['server'] = local['server']
        if sim['beam-height'] != local['beam-height']:
            logger.warning('Skipping (beam height != {}) {}'.format(
                local['beam-height'], sim['name']))
            continue
        if not claim(sim) and not args.force:
            logger.warning('Skipping (claimed) {}'.format(sim['name']))
            continue
        if args.directory and sim['directory'] != args.directory:
            logger.warning('Skipping (not specified) {}'.format(sim['name']))
        logger.warning('Starting {}'.format(sim['name']))
        await run_simulation(sim)
        upload_report(sim)


def read_local():
    with open('local.toml') as f:
        local = toml.load(f)
    assert local['server']
    assert local['beam-height'] in [0.2, 0.5, 1.0]
    return local


def configure_logging():
    formatter = logging.Formatter('%(levelname)s %(asctime)s %(message)s', '%H:%M:%S')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler('debug.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler)
    logging.getLogger().addHandler(file_handler)
    logging.getLogger().setLevel(logging.DEBUG)
    for name in ['boto3', 'botocore', 'nose', 's3transfer']:
        logging.getLogger(name).setLevel(logging.WARNING)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--default-simulation', default='simulation.defaults.toml')
    parser.add_argument('--simulations', default='simulations.toml')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--directory')
    parser.add_argument('--histories', type=float)
    parser.add_argument('--single', action='store_true')
    return parser.parse_args()


def read_simulations(args):
    with open(args.simulations) as f:
        overrides = toml.load(f)['simulations']
    with open(args.default_simulation) as f:
        defaults = toml.load(f)
    simulations = []
    for override in overrides:
        simulation = defaults.copy()
        simulation.update(override)
        if args.histories:
            simulation['total-histories'] = args.histories
        simulation['single-op'] = args.single
        simulations.append(simulation)
    return [verify_sim(sim) for sim in simulations]


def claim(simulation):
    """Clearly not thread safe, but it'll do the trick."""
    s3 = boto3.resource('s3')
    key = os.path.join(os.path.basename(simulation['directory']), 'claimed.toml')
    try:
        s3.Object('xcite-simulations', key).get()
    except BotoClientError:
        body = io.BytesIO(toml.dumps(simulation).encode('utf-8'))
        s3.Object('xcite-simulations', key).put(Body=body)
        return True
    return False


def upload_report(sim):
    s3 = boto3.client('s3')
    path = os.path.join(sim['directory'], 'report.pdf')
    slug = os.path.basename(sim['directory'])
    key = os.path.join(slug, slug + '.pdf')
    with open(path, 'rb') as f:
        s3.put_object(
            Bucket='xcite-simulations',
            Key=key,
            Body=f,
            ACL='public-read'
        )
    url = 'https://s3-us-west-2.amazonaws.com/xcite-simulations/' + key
    logger.info('Report uploaded to {}'.format(url))


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
        command = ['beamdpr', 'sample-combine', '--rate', str(rate), '-o', temp_path]
        await run_command(command + paths)
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


async def combine_doses(sim, doses):
    """
    what is the purpose here? how do we weight the arcs?
    the problem is the arc dose seems different
    but we can just flatten it out

    how do we weight the arc dose?
    """
    logger.info('Combining doses')
    print(doses)
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
        path = os.path.join(sim['directory'], '{}.3ddose'.format(stage))
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


def verify_sim(sim):
    sim['directory'] = os.path.abspath(sim['name'].replace(' - ', '-').replace(' ', '-'))
    assert 'collimator' in sim, "{} does not defined collimator".format(sim['name'])
    sim['egs-home'] = os.path.abspath(os.path.join(os.environ['HEN_HOUSE'], '../egs_home/'))
    paths = [
        sim['beamnrc-template'], sim['stationary-dose-template'],
        sim['arc-dose-template'], sim['grace'],
        os.path.join(os.environ['HEN_HOUSE'], 'pegs4', 'data', sim['pegs4'] + '.pegs4dat'),
    ]
    for path in paths:
        assert os.path.exists(path), "Could not find {}".format(path)
    floats = [
        'rmax', 'beam-width', 'beam-height', 'beam-gap',
        'target-distance', 'target-length', 'target-angle'
    ]
    ints = [
        'total-histories', 'dose-recycle', 'dose-photon-splitting'
    ]
    sim['phantom-isocenter'] = list(map(float, sim['phantom-isocenter']))
    for key in floats:
        sim[key] = float(sim[key])
    for key in ints:
        sim[key] = int(sim[key])
    return sim


async def get_templates(sim):
    stages = ['source', 'filter', 'collimator', 'dose']
    templates = {stage: template for stage, template in zip(stages, await asyncio.gather(*[
        build.build_source(sim),
        build.build_filter(sim),
        build.build_collimator(sim)
    ]))}
    with open(sim['stationary-dose-template']) as f:
        templates['stationary_dose'] = f.read()
    with open(sim['arc-dose-template']) as f:
        templates['arc_dose'] = f.read()
    return templates


async def run_simulations(sim, templates):
    y_values = generate_y(sim['target-length'], sim['beam-width'] + sim['beam-gap'], sim['reflect'])
    sim['beamlet-count'] = len(y_values)
    sim['beamlet-histories'] = sim['total-histories'] // sim['beamlet-count']
    if sim['reflect']:
        sim['beamlet-histories'] //= 2
        sim['beamlet-count'] *= 2
    sim['electrons'] = sim['beamlet-count'] * sim['beamlet-histories']
    simulations = []
    for i, y in enumerate(y_values):
        if sim['reflect']:
            index = (len(y_values) - i - 1, i + len(y_values))
        else:
            index = i
        simulations.append(simulate.simulate(sim, templates, index, y))
        if sim['single-op']:
            break
    simulations = await asyncio.gather(*simulations)


def regroup(ls):
    """
    Takes a list of dictionaries and turns it into a dictionary of lists
    """
    if not ls:
        return {}
    return {key: [item[key] for item in ls] for key in ls[0]}


def group_doses(doses, reflect):
    doses = {}
    if reflect:
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


def link_combined(combined, directory):
    for key in ['source', 'filter', 'collimator']:
        source = os.path.abspath(combined[key])
        link_name = os.path.join(directory, 'sampled_{}.egsphsp'.format(key))
        force_symlink(source, link_name)


async def generate_grace(sim, combined):
    with open(sim['grace']) as f:
        plots = toml.load(f)['plots']
    grace_plots = await grace.make_plots(combined, plots)
    os.makedirs(os.path.join(sim['directory'], 'grace'), exist_ok=True)
    for plot_type, plots in grace_plots.items():
        for plot in plots:
            for ext in ['grace', 'eps']:
                source = os.path.abspath(plot[ext])
                relpath = os.path.join('grace', plot['slug'] + '.' + ext)
                link_name = os.path.join(sim['directory'], relpath)
                plot['path'] = relpath
                force_symlink(source, link_name)


async def generate_contours(sim, combined):
    target = py3ddose.Target(
        np.array(sim['phantom-isocenter']),
        sim['lesion-diameter'] / 2
    )
    contour_futures = []
    for slug, path in combined['dose'].items():
        contour_futures.append(dose_contours.plot(sim['phantom'], path, target, slug))
    logger.info('Waiting for dose contours to finish')
    contours = await asyncio.gather(*contour_futures)
    logger.info('Regrouping contour plots')
    contour_plots = OrderedDict()
    for stage in ['stationary', 'stationary_weighted', 'arc', 'arc_weighted']:
        for contour in [c for cs in contours for c in cs]:
            if contour['output_slug'] == stage:
                contour_plots.setdefault(contour['plane'], []).append(contour)
    return contour_plots


async def run_simulation(sim):
    logger.info('Starting simulation {}'.format(sim['name']))
    start = time.time()
    templates = get_templates(sim)
    simulations = regroup(await run_simulations(sim, templates))
    logger.info('All simulations finished, combining')
    doses = group_doses(simulations['dose'], sim['reflect'])

    stages = ['source', 'filter', 'collimator', 'dose']
    combined = {stage: result for stage, result in zip(stages, await asyncio.gather(*[
        sample_combine(beamlets['source'], sim['reflect']),
        sample_combine(beamlets['filter'], sim['reflect']),
        sample_combine(beamlets['collimator'], sim['reflect']),
        combine_doses(sim, doses)
    ]))}
    logger.info('Linking combined phase space files')
    link_combined(combined, sim['directory'])

    # plots
    logger.info('Loading grace configuration')
    grace_plots = await generate_grace(sim, combined)

    # scads
    logger.info('Generating scad visualizations')
    # first we need to generate the scad using beamviz
    # then we need to use scad on it
    # we'll run this every time for now, it's not that bad


    logger.info('Starting dose contour plots')
    contour_plots = generate_contours(sim, combined)

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
        photons[stage] = sum([s[stage]['stats']['total_photons'] for s in simulations])
    data = {
        '_filter': templates['filter'],
        'collimator': templates['collimator'],
        'collimator_stats': collimator_analyzer.analyze(templates['collimator']),
        'grace_plots': grace_plots,
        'contour_plots': contour_plots,
        'skin_distance': sim['collimator']['lesion-distance'] - abs(sim['phantom-isocenter'][2]),
        'ci': conformity,
        'ts': target_to_skin,
        'electrons': sim['electrons'],
        'photons': photons
    }
    report.generate(data, sim)
    logger.info('Finished in {:.2f} seconds'.format(time.time() - start))
    logger.info('Output in {}'.format(sim['directory']))
