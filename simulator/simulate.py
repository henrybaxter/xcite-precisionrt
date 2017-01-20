import json
import logging
import os
import asyncio
import math
import hashlib
import shutil
from collections import OrderedDict

import pytoml as toml
import numpy as np
from beamviz import visualize

from . import py3ddose
from . import dose_contours
from . import build
from .utils import run_command, force_symlink, regroup
from . import beamlet
from . import screenshots
from . import dvh

from . import collimator_analyzer
from . import grace
from . import report


logger = logging.getLogger(__name__)


async def run_simulation(sim):

    try:
        shutil.rmtree(sim['directory'])
    except OSError:
        logger.warning('Could not remove {}'.format(sim['directory']))
    os.makedirs(sim['directory'])

    target = py3ddose.Target(np.array(sim['lesion']), sim['lesion-diameter'] / 2)

    # figure out the source situation
    y_values = generate_y(sim['target-length'], sim['beam-width'] + sim['beam-gap'], sim['reflect'])
    sim['beamlet-count'] = len(y_values) * (2 if sim['reflect'] else 1)
    sim['beamlet-histories'] = sim['desired-histories'] // sim['beamlet-count']
    sim['total-histories'] = sim['beamlet-histories'] * sim['beamlet-count']

    # generate the templates that will be used by every simulation
    templates = await generate_templates(sim)

    # generate all simulations
    beamlets = await run_beamlets(sim, templates, y_values)

    collimator_path = os.path.join(sim['directory'], 'collimator.egsinp')
    force_symlink(beamlets['collimator'][0]['egsinp'], collimator_path)
    scad_path = visualize.render(collimator_path, sim['lesion-diameter'])

    # combine beamlets
    phsp, doses = await asyncio.gather(*[
        combine_phsp(beamlets, sim['reflect']),
        combine_dose(sim, beamlets)
    ])

    futures = {
        'grace_plots': grace.make_plots(toml.load(open(sim['grace']))['plots'], phsp),
        'contour_plots': generate_contour_plots(doses, sim['phantom'], target),
        #'screenshots': screenshots.make_screenshots(toml.load(open(sim['screenshots']))['shots'], scad_path),
        'ci': generate_conformity(doses, target),
        'ts': generate_target_to_skin(doses, target),
    }

    photons = {}
    for key in ['source', 'filter', 'collimator']:
        photons[key] = sum(bm['stats']['total_photons'] for bm in beamlets[key]) * (2 if sim['reflect'] else 1)

    context = {
        'templates': templates,
        'collimator_stats': collimator_analyzer.analyze(templates['collimator']),
        'simulation': sim,
        'photons': photons,
        'doses': py3ddose.dose_stats(py3ddose.read_3ddose(doses['arc-weighted']), target),
        'dvh': dvh.plot_dvh(sim, py3ddose.dvh(py3ddose.read_3ddose(doses['arc-weighted']), target))
    }

    # turn futures into our context
    for key, future in futures.items():
        assert key not in context
        context[key] = await future

    # symlink the combined data, beamlets, charts, etc
    link_phsp(sim['directory'], phsp)
    link_doses(sim['directory'], doses)
    link_doselets(sim['directory'], beamlets)
    link_contours(sim['directory'], context['contour_plots'])
    link_grace(sim['directory'], context['grace_plots'])

    report.generate(sim, context)


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


async def generate_templates(sim):
    stages = ['source', 'filter', 'collimator']
    templates = dict(zip(stages, await asyncio.gather(*[
        build.build_source(sim),
        build.build_filter(sim),
        build.build_collimator(sim)
    ])))
    with open(sim['stationary-dose-template']) as f:
        templates['stationary_dose'] = f.read()
    with open(sim['arc-dose-template']) as f:
        templates['arc_dose'] = f.read()
    return templates


def link_phsp(output_dir, phsp):
    folder = os.path.join(output_dir, 'phsp')
    os.makedirs(folder, exist_ok=True)
    for key, path in phsp.items():
        source = os.path.abspath(path)
        link_name = os.path.join(folder, 'sampled_{}.egsphsp'.format(key))
        force_symlink(source, link_name)


def link_contours(output_dir, contours):
    folder = os.path.join(output_dir, 'contours')
    try:
        shutil.rmtree(folder)
    except OSError:
        pass
    shutil.copytree('contours', folder)


def link_grace(output_dir, grace_plots):
    # symlink grace plots
    folder = os.path.join(output_dir, 'grace')
    os.makedirs(folder, exist_ok=True)
    for plot_type, plots in grace_plots.items():
        for plot in plots:
            for typ in ['grace', 'eps']:
                plot['path'] = os.path.join('grace', plot['slug'])
                source = os.path.abspath(plot[typ])
                link_name = os.path.join(folder, plot['slug'] + '.' + typ)
                force_symlink(source, link_name)


def link_doses(output_dir, doses):
    folder = os.path.join(output_dir, 'dose')
    os.makedirs(folder, exist_ok=True)
    for key, path in doses.items():
        source = os.path.abspath(path)
        link_name = os.path.join(folder, '{}.3ddose.npz'.format(key))
        force_symlink(source, link_name)


def link_doselets(output_dir, beamlets):
    folder = os.path.join(output_dir, 'doselets')
    os.makedirs(folder, exist_ok=True)
    keys = ['stationary', 'arc']
    for key in keys:
        doselets = beamlets[key]
        if key == 'arc':
            doselets = flatten(doselets)
        os.makedirs(os.path.join(folder, key), exist_ok=True)
        for doselet in doselets:
            source = doselet['npz']
            if 'phimin' in doselet:
                slug = '{}-{}-{}'.format(doselet['index'], doselet['phimin'], doselet['phimax'])
            else:
                slug = str(doselet['index'])
            link_name = os.path.join(folder, key, slug + '.3ddose.npz')
            force_symlink(source, link_name)


async def generate_conformity(doses, target):
    conformity = {}
    for slug, path in doses.items():
        dose = py3ddose.read_3ddose(path)
        conformity[slug] = py3ddose.paddick(dose, target)
    return conformity


async def generate_target_to_skin(doses, target):
    target_to_skin = {}
    for slug, path in doses.items():
        dose = py3ddose.read_3ddose(path)
        target_to_skin[slug] = py3ddose.target_to_skin(dose, target)
    return target_to_skin


async def generate_contour_plots(doses, phantom, target):
    os.makedirs('contours', exist_ok=True)
    futures = []
    for slug, path in doses.items():
        print(slug, path)
        futures.append(dose_contours.plot(phantom, path, target, slug))
    contours = await asyncio.gather(*futures)
    contour_plots = OrderedDict()
    for key in doses.keys():
        for contour in [c for cs in contours for c in cs]:
            if contour['output_slug'] == key:
                contour_plots.setdefault(contour['plane'], []).append(contour)
    return contour_plots


async def combine_phsp(beamlets, reflect):
    operations = {
        'source': sample_combine(beamlets['source'], reflect),
        'filter': sample_combine(beamlets['filter'], reflect),
        'collimator': sample_combine(beamlets['collimator'], reflect)
    }
    return {name: await future for name, future in operations.items()}


async def dose_combine(doses, weights=None):
    # ok we need to take the hash of each, eg 3ddose path
    inputs = [d['3ddose'] for d in doses]
    if weights is not None:
        inputs += list(weights)
    s = 'combined' + json.dumps(inputs)
    base = hashlib.md5(s.encode('utf-8')).hexdigest()
    os.makedirs('combined', exist_ok=True)
    npz_path = os.path.join('combined', base + '.3ddose.npz')
    # dose_path = os.path.join('combined', base + '.3ddose')
    if not os.path.exists(npz_path):
        _dose = None
        _doses = []
        for dose in doses:
            if 'dose' not in dose:
                dose['dose'] = py3ddose.read_3ddose(dose['npz'])
            _doses.append(dose['dose'].doses)
            _dose = dose['dose']
        _doses = np.array(_doses)
        if weights is not None:
            result = (_doses.T * weights).T
        else:
            result = _doses
        result = py3ddose.Dose(_dose.boundaries, result.sum(axis=0), _dose.errors)
        # py3ddose.write_3ddose(dose_path, result)
        py3ddose.write_npz(npz_path, result)
    return npz_path


async def optimize_stationary(sim, doses):
    sz = len(doses)
    coeffs = np.polyfit([0, sz // 2, sz - 1], [sim['x-max'], sim['x-min'], sim['x-max']], 2)
    weights = np.polyval(coeffs, np.arange(0, sz))
    return await dose_combine(doses, weights)


async def optimize_arc(sim, doses):
    futures = []
    for _doses in doses:
        print('optimize_arc step')
        sz = len(_doses)
        coeffs = np.polyfit([0, sz // 2, sz - 1], [sim['x-max'], sim['x-min'], sim['x-max']], 2)
        weights = np.polyval(coeffs, np.arange(0, sz))
        futures.append(dose_combine(_doses, weights))
    paths = await asyncio.gather(*futures)
    sz = len(paths)
    coeffs = np.polyfit([0, sz // 2, sz - 1], [sim['arc-max'], sim['arc-min'], sim['arc-max']], 2)
    weights = np.polyval(coeffs, np.arange(0, sz))
    base = hashlib.md5(json.dumps(paths).encode('utf-8')).hexdigest()
    path = 'combined/{}.3ddose'.format(base)
    py3ddose.weight_3ddose(paths, path, weights)
    py3ddose.read_3ddose(path)
    return path + '.npz'


def flatten(ls):
    result = []
    for l in ls:
        result.extend(l)
    return result


async def combine_dose(sim, beamlets):
    return {
        'stationary': await dose_combine(beamlets['stationary']),
        'arc': await dose_combine(flatten(beamlets['arc'])),
        'stationary-weighted': await optimize_stationary(sim, beamlets['stationary']),
        'arc-weighted': await optimize_arc(sim, beamlets['arc'])
    }


async def run_beamlets(sim, templates, y_values):
    beamlets = []
    for i, y in enumerate(y_values):
        if sim['reflect']:
            index = (len(y_values) - i - 1, i + len(y_values))
        else:
            index = i
        beamlets.append(beamlet.simulate(sim, templates, index, y))
        if i == sim['operations']:
            break
    simulations = regroup(await asyncio.gather(*beamlets))
    if sim['reflect']:
        # dose calculations were put into tuples
        # so flatten them, respecting position
        for key in ['stationary', 'arc']:
            flat = []
            for from_reflected, from_original in simulations[key]:
                flat.insert(0, from_reflected)
                flat.append(from_original)
            simulations[key] = flat
    return simulations
