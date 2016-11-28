import datetime
import time
import shutil
import argparse
import sys
from subprocess import run as run_process, PIPE
import os
import math
import hashlib
import json
import functools
import logging

import boto3
from scipy.optimize import fsolve
from pathos.pools import ProcessPool as Pool
from pathos.helpers import cpu_count

import latexmake
import egsinp
import grace
import interpolation

logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

# fix dyld problem:
# $ install_name_tool -change ../egs++/dso/osx/libiaea_phsp.dylib /Users/henry/projects/EGSnrc/HEN_HOUSE/egs++/dso/osx/libiaea_phsp.dylib /Users/henry/projects/EGSnrc/egs_home/bin/osx/BEAM_TUMOTRAK


class Output(object):

    def __init__(self):
        pass
        # self.bucket = boto3.resource('s3').Bucket('xcite-simulations')

    def set_dir(self, directory):
        self.directory = directory

    def send_contents(self, contents, filename):
        logger.info('Saving {}'.format(filename))
        path = os.path.join(self.directory, filename)
        open(path, 'w').write(contents)
        # self.upload(path, os.path.join(self.directory, filename))

    def send_file(self, path, filename):
        new_path = os.path.join(self.directory, filename)
        logger.info('Saving {} to {}'.format(path, new_path))
        shutil.copy(path, new_path)
        # self.upload(path, os.path.join(self.directory, filename))

    def upload(self, path, key):
        if os.path.getsize(path) > 1024 * 1024 * 10:
            logger.warning('Skipping {} it is too big'.format(key))
        else:
            self.bucket.upload_file(path, key)
output = Output()


def parse_args():
    stages = ['source', 'filter', 'collimator']
    parser = argparse.ArgumentParser()

    # common
    parser.add_argument('--egsinp-template', default='template.egsinp')
    parser.add_argument('--pegs4', default='allkV')
    parser.add_argument('--name', required=True)
    parser.add_argument('--overwrite', '-f', action='store_true')
    parser.add_argument('--rmax', type=float, default=50.0, help='Max extent of all component modules')

    # source arguments
    parser.add_argument('--beam-width', type=float, help='Beam width (y) in cm', default=0.2)
    parser.add_argument('--beam-height', type=float, help='Beam height (z) in cm', default=0.5)
    parser.add_argument('--beam-distance', type=float, help='Beam axis of rotation to target in cm', default=10.0)
    parser.add_argument('--target-length', type=float, help='Length of target in cm', default=75.0)
    parser.add_argument('--target-angle', default=45.0, type=float, help='Angle of reflection target')
    increments = ['dy_simulations', 'dy_gap', 'dtheta', 'dflare', 'dlinear']
    parser.add_argument('--increment', choices=increments, default='dy_gap')
    parser.add_argument('--gap', type=float, default=0.0, help='Gap between incident beams (only valid if dy selected')
    parser.add_argument('--histories', type=int, default=int(1e9), help='Divided among source beamlets')

    # utils
    parser.add_argument('--clean', action='store_true', help='Remove all directories, forces rebuild')
    parser.add_argument('--build', '-b', dest='builds', choices=stages + ['all'], action='append', default=[], help='Build BEAM_* executable')

    # collimator
    #   generated
    parser.add_argument('--collimator-length', type=float, default=12.0, help='Length of collimator')
    parser.add_argument('--interpolating-blocks', type=int, default=2, help='Number of interpolating blocks to use for collimator')
    parser.add_argument('--phantom-target-distance', type=float, default=75.0, help='Distance from end of collimator to phantom target')
    parser.add_argument('--beam-weighting', action='store_true', help='Weight beams by r^2/r\'^2')
    parser.add_argument('--septa-width', type=float, default=0.05, help='Septa width')
    parser.add_argument('--hole-width', type=float, default=0.2, help='Minor size of hexagon')
    #   given
    parser.add_argument('--collimator', help='Input egsinp path or use stamped values')
    #   preprocess filter phase space
    parser.add_argument('--rotate', action='store_true', help='Rotate phase space files before collimator?')

    # dose
    parser.add_argument('--phantom', default='16cmcylinder2mmvoxel.egsphant', help='.egsphant file')
    parser.add_argument('--dos-egsinp', default='dosxyz_input_template.egsinp', help='.egsinp for dosxyznrc')
    parser.add_argument('--dose-histories', default=50000000, type=int, help='Histories for dosxyznrc')

    args = parser.parse_args()
    args.output_dir = args.name.replace(' ', '-')
    output.set_dir(args.output_dir)
    if os.path.exists(args.output_dir):
        if args.overwrite:
            remove_folders([args.output_dir])
        else:
            logger.warning('Output path {} exists already'.format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    if 'all' in args.builds:
        args.builds = set(stages)
    args.egs_home = os.path.abspath(os.path.join(os.environ['HEN_HOUSE'], '../egs_home/'))
    logger.info('egs_home is {}'.format(args.egs_home))

    if args.collimator:
        # always gets rebuilt
        collimator_folder = 'BEAM_CLMTX'
    else:
        collimator_folder = 'BEAM_CLMT{}'.format(args.interpolating_blocks)

    args.folders = {
        'source': os.path.join(args.egs_home, 'BEAM_RFLCT'),
        'filter': os.path.join(args.egs_home, 'BEAM_FILTR'),
        'collimator': os.path.join(args.egs_home, collimator_folder),
    }
    args.simulation_properties = {
        'common': {
            'name': args.name,
            'pegs4': args.pegs4,
            'scoring_zone_size': args.rmax
        },
        'source': {
            'beam_weighting': args.beam_weighting,
            'rmax': args.rmax,
            'histories': args.histories,
            'beam_distance': args.beam_distance,
            'beam_height': args.beam_height,
            'beam_width': args.beam_width,
            'length': args.target_length,
            'angle': args.target_angle,
            'type': 'Reflection',
            'beam_merge_strategy': 'Translation and Combination'
        },
        'filter': {
            'rmax': args.rmax,
            'slabs': []
        },
        'collimator': {
            'rmax': args.rmax,
            'template': args.collimator,
            'length': None,
            'septa_width': args.septa_width,
            'hole_width': args.hole_width,
            'regions_per_block': None,
            'interpolating_blocks': None
        },
    }
    return args


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def write_specmodule(egs_home, name, cms):
    logger.info('Writing spec module for {}'.format(name))
    types = []
    identifiers = []
    for cm in cms:
        types.append(cm['type'])
        identifiers.append(cm['identifier'])
    types_line = ' CM names:  {}'.format(' '.join(types))
    identifiers_line = ' Identifiers:  {}'.format(' '.join(identifiers))
    path = os.path.join(egs_home, 'beamnrc/spec_modules', '{}.module'.format(name))
    open(path, 'w').write('\n'.join([types_line, identifiers_line]))
    logger.info('Spec module written')


def beam_build(egs_home, name, cms):
    logger.info('Building {}'.format(name))
    start = time.time()
    write_specmodule(egs_home, name, cms)
    command = ['beam_build.exe', name]
    logger.info('Running beam_build.exe')
    result = run_process(command, stdout=PIPE, stderr=PIPE)
    assert result.returncode == 0
    logger.info('Running make')
    beam_folder = os.path.join(egs_home, 'BEAM_{}'.format(name))
    result = run_process(['make'], cwd=beam_folder, stdout=PIPE, stderr=PIPE)
    assert result.returncode == 0
    elapsed = time.time() - start
    logger.info('{} built in {} seconds'.format(name, elapsed))


def generate_dy(target_length, beam_width, spacing):
    """By spacing, use equal changes in incidence y coordinate"""
    logger.info('Generating y values by dy')
    gap = spacing - beam_width
    logger.info('Spacing is {:3f} which means gaps of {:3f}'.format(spacing, gap))
    offset = spacing / 2
    y = offset
    ymax = target_length / 2
    i = 0
    result = []
    while y < ymax:
        result.append(y)
        i += 1
        y = i * spacing + offset
    return result


def generate_dy_simulations(target_length, beam_width, simulations):
    """By simulations, use equal changes in incidence y coordinate"""
    spacing = target_length / 2 / simulations
    return generate_dy(target_length, beam_width, spacing)


def generate_dy_gaps(target_length, beam_width, gap):
    """By gap, use equal changes in incidence y coordinate"""
    spacing = beam_width + gap
    return generate_dy(target_length, beam_width, spacing)


def generate_dtheta(target_length, beam_distance, simulations):
    """By number of simulations, use equal changes in incidence angle"""
    logger.info('Generating y values by dtheta')
    ymax = target_length / 2
    max_theta = math.atan(ymax / beam_distance)
    dtheta = max_theta / simulations
    offset = dtheta / 2
    theta = offset
    i = 0
    result = []
    while theta < max_theta:
        y = beam_distance * math.tan(theta)
        result.append(y)
        i += 1
        theta = i * dtheta + offset
    return result


def generate_dflare(target_length, beam_distance, beam_width):
    """Using primary variables, solve for beam flares touching"""
    logger.info('Genearting y values by dflare')
    ymax = target_length / 2
    y = 0
    theta = 0
    result = []
    while True:
        def f(next_theta):
            # current y + half previous flare + half next flare = next y
            flare = beam_width / math.cos(theta)
            next_flare = beam_width / math.cos(next_theta)
            next_y = beam_distance * math.tan(next_theta)
            return y + flare / 2 + next_flare / 2 - next_y
        theta = fsolve(f, theta)[0]
        y = beam_distance * math.tan(theta)
        if y < ymax:
            result.append(y)
        else:
            break
    return result


def generate_dlinear(target_length, beam_width, simulations):
    """Just linearly slowly decrease the gap from max to 0 over simulations"""
    logger.info('Generating y values by dlinear with {} simulations'.format(simulations))
    max_spacing = 2
    offset = max_spacing / 2
    y = offset
    ymax = target_length / 2
    slope = ymax / max_spacing
    i = 0
    result = []
    while y < ymax:
        result.append(y)
        i += 1
        y = offset + slope * i
    return result


def get_egsinp(path):
    logger.info('Reading egsinp template')
    try:
        text = open(path).read()
    except IOError:
        logger.error('Could not open template {}'.format(path))
        sys.exit(1)
    try:
        template = egsinp.parse_egsinp(text)
    except egsinp.ParseError as e:
        logger.error('Could not parse template {}: {}'.format(path, e))
        sys.exit(1)
    return template


def dose_simulations(folder, pegs4, simulations):
    to_simulate = simulations
    #for simulation in simulations:
    #    if not os.path.exists(simulation['dose']):
    #        to_simulate.append(simulation)
    logger.info('Reusing {} and running {} dose calculations'.format(len(simulations) - len(to_simulate), len(to_simulate)))
    dose = functools.partial(dose_simulation, folder, pegs4)
    pool = Pool(cpu_count() - 1)
    pool.map(dose, to_simulate)


def dose_simulation(folder, pegs4, simulation):
    try:
        os.remove(simulation['dose'])
    except IOError:
        pass
    command = ['dosxyznrc', '-p', pegs4, '-i', simulation['egsinp']]
    logger.info('Running "{}"'.format(' '.join(command)))
    result = run_process(command, stdout=PIPE, stderr=PIPE, cwd=folder)
    command_output = result.stdout.decode('utf-8') + result.stderr.decode('utf-8')
    if result.returncode != 0 or 'ERROR' in command_output:
        logger.error('Could not run dosxyz on {}'.format(simulation['egsinp']))
        logger.error(result.args)
        logger.error(command_output)
    #logger.info(result)
    egslst = os.path.join(folder, simulation['egsinp'].replace('.egsinp', '.egslst'))
    logger.info('Writing to {}'.format(egslst))
    open(egslst, 'w').write(command_output)
    if 'Warning' in command_output:
        logger.info('Warning in {}'.format(egslst))


def beam_simulations(folder, pegs4, simulations):
    to_simulate = []
    translate_y = False
    for simulation in simulations:
        translate_y = translate_y or 'translate_y' in simulation
        if not os.path.exists(simulation['phsp']):
            to_simulate.append(simulation)
    logger.info('Reusing {} and running {} simulations'.format(len(simulations) - len(to_simulate), len(to_simulate)))
    simulate = functools.partial(beam_simulation, folder, pegs4)
    pool = Pool(cpu_count() - 1)
    pool.map(simulate, to_simulate)

    if translate_y:
        logger.info('Translating source phase space files')
        pool = Pool(cpu_count() - 1)
        translate = functools.partial(beamdpr_translate, folder)
        pool.map(translate, to_simulate)

    """
    logger.info('Checking sizes of phase space files')
    sizes = []
    for simulation in simulations:
        sizes.append(os.path.getsize(simulation['phsp']))
    mean = statistics.mean(sizes)
    stdev = statistics.pstdev(sizes)
    for size, simulation in zip(sizes, simulations):
        if abs(size - mean) > stdev * 3:
            logger.warning('Simulation {} is an unusual size'.format(simulation['phsp']))
    """


def beam_simulation(folder, pegs4, simulation):
    """Always remove the target phase space file"""
    try:
        os.remove(simulation['phsp'])
        logger.info('Removed old phase space file {}'.format(simulation['phsp']))
    except IOError:
        pass
    executable = os.path.basename(os.path.normpath(folder))
    command = [executable, '-p', pegs4, '-i', simulation['egsinp']]
    result = None
    logger.info('Running "{}"'.format(' '.join(command)))
    result = run_process(command, stdout=PIPE, stderr=PIPE, cwd=folder)
    command_output = result.stdout.decode('utf-8') + result.stderr.decode('utf-8')
    if result.returncode != 0 or 'ERROR' in command_output:
        logger.error('Could not run simulation on {}'.format(simulation['egsinp']))
        logger.error(result.args)
        logger.error(command_output)


def sample_combine(beamlets, desired=10000000):
    logger.info('Sampling and combining {} beamlets'.format(len(beamlets)))
    paths = [beamlet['phsp'] for beamlet in beamlets]
    particles = sum([beamlet['stats']['total_particles'] for beamlet in beamlets])
    rate = math.ceil(particles / desired)
    logger.info('Found {} particles, want {}, sample rate is {}'.format(particles, desired, rate))
    s = 'rate={}'.format(rate) + ''.join([beamlet['hash'].hexdigest() for beamlet in beamlets])
    md5 = hashlib.md5(s.encode('utf-8'))
    os.makedirs('combined', exist_ok=True)
    phsp = 'combined/{}.egsphsp1'.format(md5.hexdigest())
    if os.path.exists(phsp):
        logger.info('Combined beamlets file {} already exists'.format(phsp))
        return phsp
    logger.info('Combining {} beamlets into {}'.format(len(beamlets), phsp))
    command = ['beamdpr', 'sample-combine', '--rate', str(rate), '-o', phsp] + paths
    result = run_process(command, stdout=PIPE, stderr=PIPE)
    if result.returncode != 0:
        logger.error('Command failed: "{}"'.format(' '.join(command)))
        logger.error(result.stdout.decode('utf-8'))
        logger.error(result.stderr.decode('utf-8'))
        sys.exit(1)
    logger.info('Randomizing')
    command = ['beamdpr', 'randomize', phsp]
    result = run_process(command, stdout=PIPE, stderr=PIPE)
    if result.returncode != 0:
        logger.error('Command failed: "{}"'.format(' '.join(command)))
        logger.error(result.stdout.decode('utf-8'))
        logger.error(result.stderr.decode('utf-8'))
        sys.exit(1)
    return phsp


def beamdpr_stats(path):
    command = ['beamdpr', 'stats', '--format=json', path]
    result = run_process(command, stdout=PIPE, stderr=PIPE)
    if result.returncode != 0:
        logger.error('Command failed: "{}"'.format(' '.join(command)))
        logger.error(result.stdout.decode('utf-8'))
        logger.error(result.stderr.decode('utf-8'))
        sys.exit(1)
    return json.loads(result.stdout.decode('utf-8'))


def beamlet_stats(beamlets):
    logger.info('Gathering stats for {} beamlets'.format(len(beamlets)))
    for beamlet in beamlets:
        beamlet['stats'] = beamdpr_stats(beamlet['phsp'])
    return beamlets


def beamdpr_translate(folder, item):
    command = ['beamdpr', 'translate', '-i', item['phsp'], '-y', '(' + str(item['translate_y']) + ')']
    result = None
    try:
        result = run_process(command, stdout=PIPE, stderr=PIPE, cwd=folder)
        assert result.returncode == 0
    except Exception as e:
        logger.error('Could not translate {}: {}'.format(item['phsp'], e))
        eprint(result)
        sys.exit(1)


def generate_y(args):
    if args.increment == 'dy_gap':
        y_values = generate_dy_gaps(args.target_length, args.beam_width, args.gap)
    elif args.increment == 'dy_simulations':
        y_values = generate_dy_simulations(args.target_length, args.beam_width, args.simulations)
    elif args.increment == 'dtheta':
        y_values = generate_dtheta(args.target_length, args.beam_distance, args.simulations)
    elif args.increment == 'dflare':
        y_values = generate_dflare(args.target_length, args.beam_distance, args.beam_width)
    elif args.increment == 'dlinear':
        y_values = generate_dlinear(args.target_length, args.beam_width, args.simulations)
    else:
        raise ValueError('Unexpected increment {}'.format(args.increment))
    # generate symmetric y values
    for y in y_values[:]:
        y_values.insert(0, -y)
    return y_values


def generate_source(args):
    logger.info('Generating source')
    y_values = generate_y(args)
    template = get_egsinp(args.egsinp_template)

    # rebuild mortran if requested
    if 'source' in args.builds or not os.path.exists(args.folders['source']):
        beam_build(args.egs_home, 'RFLCT', template['cms'])
    template['ncase'] = args.histories // len(y_values)
    template['ybeam'] = args.beam_width / 2
    template['zbeam'] = args.beam_height / 2
    logger.info('Using {} histories per beamlet for a total of {} source histories'.format(template['ncase'], template['ncase'] * len(y_values)))
    xtube = template['cms'][0]
    xtube['rmax_cm'] = args.rmax
    xtube['anglei'] = args.target_angle

    logger.info('Creating egsinp files')
    beamlets = []
    simulations = []
    first = True
    histories = 0
    for y in y_values:
        if args.beam_weighting:
            weight = 1 + (y * y) / (args.phantom_target_distance * args.phantom_target_distance)
            template['ncase'] = int(args.histories / len(y_values) * weight)
            logger.info('Setting beam at {} to {} histories'.format(y, template['ncase']))
            histories += template['ncase']
        theta = math.atan(y / args.beam_distance)
        cos_x = -math.cos(theta)
        cos_y = math.copysign(math.sqrt(1 - cos_x * cos_x), y)
        template['uinc'] = cos_x
        template['vinc'] = cos_y
        egsinp_str = egsinp.unparse_egsinp(template)
        md5 = hashlib.md5(egsinp_str.encode('utf-8'))
        base = md5.hexdigest()
        inp = '{}.egsinp'.format(base)
        inp_path = os.path.join(args.folders['source'], inp)
        open(inp_path, 'w').write(egsinp_str)
        if first:
            output.send_contents(egsinp_str, 'source.egsinp')
        phsp = os.path.join(args.folders['source'], '{}.egsphsp1'.format(base))
        simulations.append({
            'egsinp': inp,
            'phsp': phsp,
            'translate_y': y
        })
        beamlets.append({
            'phsp': phsp,
            'hash': md5
        })
        first = False
    args.simulation_properties['source']['histories'] = histories
    beam_simulations(args.folders['source'], args.pegs4, simulations)

    return beamlets


def filter_source(beamlets, args):
    logger.info('Filtering')
    template = get_egsinp(args.egsinp_template)
    template['cms'] = [
        {
            'type': 'SLABS',
            'identifier': 'FLTR',
            'rmax_cm': args.rmax,
            'title': 'FLTR',
            'zmin_slabs': 0.01,
            'slabs': [
                {
                    'zthick': 0.1,
                    'ecut': 0.521,
                    'pcut': 0.001,
                    'dose_zone': 0,
                    'iregion_to_bit': 0,
                    'esavein': 0,
                    'medium': 'Al_516kV'
                },
                {
                    'zthick': 0.3,
                    'ecut': 0.521,
                    'pcut': 0.001,
                    'dose_zone': 0,
                    'iregion_to_bit': 0,
                    'esavein': 0,
                    'medium': 'H2O_516kV'
                },
                {
                    'zthick': 0.05,
                    'ecut': 0.521,
                    'pcut': 0.001,
                    'dose_zone': 0,
                    'iregion_to_bit': 0,
                    'esavein': 0,
                    'medium': 'steel304L_521kV'
                },

            ]
        }
    ]
    for i, slab in enumerate(template['cms'][0]['slabs']):
        description = 'Slab {} is {} cm of {}'.format(i + 1, slab['zthick'], slab['medium'])
        args.simulation_properties['filter']['slabs'].append(description)
        logger.info(description)
    template['isourc'] = '21'
    template['iqin'] = '0'
    for k in ['nrcycl', 'iparallel', 'parnum', 'isrc_dbs', 'rsrc_dbs', 'ssdrc_dbs', 'zsrc_dbs']:
        template[k] = '0'
    template['init_icm'] = 1
    if 'filter' in args.builds or not os.path.exists(args.folders['filter']):
        beam_build(args.egs_home, 'FILTR', template['cms'])
    filtered_beamlets = []
    simulations = []
    first = True
    for beamlet in beamlets:
        template['ncase'] = beamlet['stats']['total_particles']
        template['spcnam'] = os.path.join('../', 'BEAM_RFLCT', os.path.basename(beamlet['phsp']))
        egsinp_str = egsinp.unparse_egsinp(template)
        md5 = beamlet['hash'].copy()
        md5.update(egsinp_str.encode('utf-8'))
        base = md5.hexdigest()
        inp = '{}.egsinp'.format(base)
        inp_path = os.path.join(args.folders['filter'], inp)
        open(inp_path, 'w').write(egsinp_str)
        if first:
            output.send_contents(egsinp_str, 'filter.egsinp')
        phsp = os.path.join(args.folders['filter'], '{}.egsphsp1'.format(base))
        simulations.append({
            'egsinp': inp,  # filename
            'phsp': phsp,  # full path
        })
        filtered_beamlets.append({
            'phsp': phsp,
            'hash': md5
        })
        first = False
    beam_simulations(args.folders['filter'], args.pegs4, simulations)

    return filtered_beamlets


def rotate(beamlets):
    logger.info("Rotating beamlets")
    for beamlet in beamlets:
        beamlet['hash'].update('zrotatep=90'.encode('utf-8'))
        filename = beamlet['hash'].hexdigest() + '.egsphsp1'
        phsp = os.path.join(os.path.dirname(beamlet['phsp']), filename)
        if not os.path.exists(phsp):
            command = ['beamdpr', 'rotate', beamlet['phsp'], phsp, '-a', str(math.pi / 2)]
            result = run_process(command, stdout=PIPE, stderr=PIPE)
            if result.returncode != 0:
                logger.error('Command failed: "{}"'.format(' '.join(command)))
                logger.error(result.stdout.decode('utf-8'))
                logger.error(result.stderr.decode('utf-8'))
                sys.exit(1)
        beamlet['phsp'] = phsp
    return beamlets


def collimate(beamlets, args):
    logger.info('Collimating')
    template = get_egsinp(args.egsinp_template)
    template['cms'] = []
    if args.collimator == 'henry-1':
        kwargs = {
            'length': args.collimator_length,
            'blocks': args.interpolating_blocks,
            'septa': args.septa_width,
            'size': args.hole_width
        }
        for i, block in enumerate(interpolation.make_hblocks(**kwargs)):
            cm = {
                'type': 'BLOCK',
                'identifier': 'BLCK{}'.format(i),
                'rmax_cm': args.rmax,
                'title': 'BLCK{}'.format(i),
                'zmin': block['zmin'],
                'zmax': block['zmax'],
                'zfocus': args.phantom_target_distance + args.collimator_length,
                'xpmax': args.rmax,
                'ypmax': args.rmax,
                'xnmax': -args.rmax,
                'ynmax': -args.rmax,
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
            template['cms'].append(cm)
    elif args.collimator:
        collimator = get_egsinp(args.collimator)
        zoffset = None
        for cm in collimator['cms']:
            if cm['type'] != 'BLOCK':
                continue
            if zoffset is None:
                zoffset = cm['zmin']
            cm['zmin'] -= zoffset
            cm['zmax'] -= zoffset
            cm['zfocus'] -= zoffset
            cm['rmax_cm'] = args.rmax
            template['cms'].append(cm)
    else:
        for i, block in enumerate(interpolation.make_blocks(args.collimator_length, args.interpolating_blocks)):
            cm = {
                'type': 'BLOCK',
                'identifier': 'BLCK{}'.format(i),
                'rmax_cm': args.rmax,
                'title': 'BLCK{}'.format(i),
                'zmin': block['zmin'],
                'zmax': block['zmax'],
                'zfocus': args.phantom_target_distance + args.collimator_length,
                'xpmax': block['xpmax'],
                'ypmax': block['ypmax'],
                'xnmax': block['xnmax'],
                'ynmax': block['ynmax'],
                'air_gap': {
                    'ecut': 0,
                    'pcut': 0,
                    'dose_zone': 0,
                    'iregion_to_bit': 0
                },
                'opening': {
                    'ecut': 0,
                    'pcut': 0,
                    'dose_zone': 0,
                    'iregion_to_bit': 0,
                    'medium': 'Air_516kVb'
                },
                'block': {
                    'ecut': 0,
                    'pcut': 0,
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
            template['cms'].append(cm)
    # print(len(template['cms']))
    template['isourc'] = '21'
    template['iqin'] = '0'
    template['default_medium'] = 'Air_516kV'
    template['nsc_planes'] = '1'
    template['scoring_planes'] = [{
        'cm': len(template['cms']),  # the LAST block of the collimator
        'mzone_type': 1,
        'nsc_zones': 1,
        'zones': [args.rmax]
    }]
    for k in ['init_icm', 'nrcycl', 'iparallel', 'parnum', 'isrc_dbs', 'rsrc_dbs', 'ssdrc_dbs', 'zsrc_dbs']:
        template[k] = '0'
    # always rebuild custom collimators!
    if 'collimator' in args.builds or not os.path.exists(args.folders['collimator']):
        beam_build(args.egs_home, os.path.basename(args.folders['collimator']).replace('BEAM_', ''), template['cms'])
    template['init_icm'] = 1
    for cm in template['cms']:
        if cm['type'] != 'BLOCK':
            raise ValueError('Unexpected collimator type {}'.format(cm['type']))
    args.simulation_properties['collimator']['interpolating_blocks'] = len(template['cms'])
    args.simulation_properties['collimator']['length'] = template['cms'][-1]['zmax'] - template['cms'][0]['zmin']
    args.simulation_properties['collimator']['regions_per_block'] = len(template['cms'][0]['regions'])
    simulations = []
    collimated_beamlets = []
    first = True
    for beamlet in beamlets:
        template['ncase'] = beamlet['stats']['total_particles']
        template['spcnam'] = '../BEAM_FILTR/' + os.path.basename(beamlet['phsp'])
        egsinp_str = egsinp.unparse_egsinp(template)
        md5 = beamlet['hash'].copy()
        md5.update(egsinp_str.encode('utf-8'))
        base = md5.hexdigest()
        inp = '{}.egsinp'.format(base)
        inp_path = os.path.join(args.folders['collimator'], inp)
        open(inp_path, 'w').write(egsinp_str)
        if first:
            output.send_contents(egsinp_str, 'collimator.egsinp')
        phsp = os.path.join(args.folders['collimator'], '{}.egsphsp1'.format(base))
        simulations.append({
            'egsinp': inp,  # filename
            'phsp': phsp,  # full path
        })
        collimated_beamlets.append({
            'phsp': phsp,
            'hash': md5
        })
        first = False
    beam_simulations(args.folders['collimator'], args.pegs4, simulations)
    # egslst = os.path.join(args.folders['collimator'], simulations[0]['egsinp'].replace('.egsinp', '.egslst'))
    # output.send_file(egslst, 'collimator.egslst')

    return collimated_beamlets


def dose(beamlets, args):
    logger.info('Dosing')
    template = open(args.dos_egsinp).read()
    directory = os.path.join(args.egs_home, 'dosxyznrc')
    dose_contributions = []
    simulations = []
    first = True
    for beamlet in beamlets:
        kwargs = {
            'egsphant_path': os.path.join(SCRIPT_DIR, args.phantom),
            'phsp_path': beamlet['phsp'],
            'ncase': beamlet['stats']['total_particles']
        }
        logger.info('Setting up dose run with {} histories'.format(kwargs['ncase']))
        egsinp_str = template.format(**kwargs)
        md5 = beamlet['hash'].copy()
        md5.update(egsinp_str.encode('utf-8'))
        base = md5.hexdigest()
        inp = '{}.egsinp'.format(base)
        inp_path = os.path.join(directory, inp)
        open(inp_path, 'w').write(egsinp_str)
        if first:
            output.send_contents(egsinp_str, 'dosxyz.egsinp')
        dose_filename = '{}.3ddose'.format(base)
        dose_path = os.path.join(directory, dose_filename)
        simulations.append({
            'egsinp': inp,
            'dose': dose_path
        })
        dose_contributions.append({
            'dose': dose_path,
            'hash': md5
        })
        first = False
    dose_simulations(directory, args.pegs4, simulations)
    return dose_contributions


def grace_plot(base_dir, phsp_paths, args):
    os.makedirs(base_dir, exist_ok=True)
    for stage in ['source', 'filter', 'collimator']:
        phsp_path = phsp_paths[stage]
        kwargs = {'extents': {
            'xmin': -args.beam_width * 2,
            'xmax': args.beam_width * 2,
            'ymin': -args.target_length / 2,
            'ymax': args.target_length / 2
        }}
        if args.rotate:
            xmin = kwargs['extents']['xmin']
            kwargs['extents']['xmin'] = kwargs['extents']['ymin']
            kwargs['extents']['ymin'] = xmin
            xmax = kwargs['extents']['xmax']
            kwargs['extents']['xmax'] = kwargs['extents']['ymax']
            kwargs['extents']['ymax'] = xmax
        path = os.path.join(base_dir, '{}_xy.grace'.format(stage))
        if not os.path.exists(path):
            grace.xy(phsp_path, path, **kwargs)
            grace.eps(path)
        x_path = os.path.join(base_dir, '{}_energy_fluence_x.grace'.format(stage))
        y_path = os.path.join(base_dir, '{}_energy_fluence_y.grace'.format(stage))
        if not os.path.exists(x_path):
            _kwargs = kwargs.copy()
            # per Magdalena's wishes
            _kwargs['axis'] = 'x'
            grace.energy_fluence_vs_position(phsp_path, x_path, **_kwargs)
            grace.eps(x_path)
        if not os.path.exists(y_path):
            _kwargs = kwargs.copy()
            _kwargs['axis'] = 'y'
            grace.energy_fluence_vs_position(phsp_path, y_path, **_kwargs)
            grace.eps(y_path)
        path = os.path.join(base_dir, '{}_spectral.grace'.format(stage))
        if not os.path.exists(path):
            grace.spectral_distribution(phsp_path, path, **kwargs)
            grace.eps(path)
        path = os.path.join(base_dir, '{}_angular.grace'.format(stage))
        if not os.path.exists(path):
            grace.angular(phsp_path, path, **kwargs)
            grace.eps(path)


def remove_folders(folders):
    for folder in folders:
        logger.info('Removing directory {}'.format(folder))
        try:
            shutil.rmtree(folder)
        except IOError:
            pass


def latex_itemize_properties(properties):
    stages = ['source', 'filter', 'collimator']
    units = {
        'beam_distance': 'cm',
        'beam_width': 'cm',
        'beam_height': 'cm',
        'length': 'cm',
        'scoring_zone_size': 'cm'
    }
    lines = []

    def latex_clean(value):
        return value.replace('_', '\\_').replace('{', '\\{').replace('}', '\\}')
    for stage in stages:
        lines.append('\t\item {}'.format(stage.capitalize()))
        lines.append('\t\\begin{itemize}')
        for key, value in sorted(properties[stage].items()):
            if not value:
                continue
            if key == 'slabs':
                lines.append('\t\t\item Slabs')
                lines.append('\t\t\\begin{enumerate}')
                for description in value:
                    lines.append('\t\t\t\item {}'.format(latex_clean(description)))
                lines.append('\t\t\\end{enumerate}')
            else:
                value = str(value)
                if key in units:
                    value += ' {}'.format(units[key])
                key = key.replace('_', ' ').capitalize()
                lines.append('\t\t\item {}: {}'.format(key, latex_clean(value)))
        lines.append('\t\end{itemize}')
    return '\n'.join(lines)


def itemize_photons(beamlets):
    lines = []
    previous = args.simulation_properties['source']['histories']
    lines.append('\t\item {}: {}'.format('Incident electrons', str(previous)))
    for stage in ['source', 'filter', 'collimator']:
        photons = sum([beamlet['stats']['total_photons'] for beamlet in beamlets[stage]])
        if previous:
            rate = previous / photons
            text = '{} photons (reduced by a factor of {:.2f})'.format(photons, rate)
        else:
            text = '{} photons'.format(photons)
        lines.append('\t\item {}: {}'.format(stage.capitalize(), text))
        previous = photons
    overall = args.simulation_properties['source']['histories'] / previous
    lines.append('\t\item Overall efficiency: {:.2f} electrons to generate one photon'.format(overall))
    return '\n'.join(lines)


def configure_logging():
    formatter = logging.Formatter('%(levelname)s %(asctime)s.%(msecs)03d %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    now = datetime.datetime.now()
    filename = 'simulate.{}.log'.format(datetime.datetime.strftime(now, '%Y-%m-%d.%H:%M:%S'))
    file_handler = logging.FileHandler(filename)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler)
    logging.getLogger().addHandler(file_handler)
    logging.getLogger().setLevel(logging.DEBUG)

if __name__ == '__main__':
    start = time.time()
    configure_logging()
    logger.info('Starting')
    args = parse_args()
    if args.clean:
        remove_folders(args.folders.values())
        sys.exit()
    skip_args = ['clean', 'builds', 'plots', 'stages']
    logger.info('Arguments: \n{}'.format('\n'.join(['\t{}: {}'.format(k, v) for k, v in sorted(args.__dict__.items()) if k not in skip_args])))
    phsp = {}
    beamlets = {}
    beamlets['source'] = beamlet_stats(generate_source(args))
    if args.rotate:
        beamlets['source'] = rotate(beamlets['source'])
    phsp['source'] = sample_combine(beamlets['source'])
    output.send_file(phsp['source'], 'sampled_source.egsphsp1')
    # output.send_file(phsp['source'].replace('.egsphsp1', '.egslst'), 'source.egslst')

    beamlets['filter'] = beamlet_stats(filter_source(beamlets['source'], args))
    phsp['filter'] = sample_combine(beamlets['filter'])
    output.send_file(phsp['filter'], 'sampled_filter.egsphsp1')
    # output.send_file(phsp['filter'].replace('.egsphsp1', '.egslst'), 'filter.egslst')

    beamlets['collimator'] = beamlet_stats(collimate(beamlets['filter'], args))
    phsp['collimator'] = sample_combine(beamlets['collimator'], desired=100000000)
    output.send_file(phsp['collimator'], 'sampled_collimator.egsphsp1')
    # output.send_file(phsp['collimator'].replace('.egsphsp1', '.egslst'), 'collimator.egslst')

    dose_contributions = dose(beamlets['collimator'], args)

    # now we take the md5 of the args? collimated beamlets.
    plots = grace_plot(args.output_dir, phsp, args)
    template = open('template.tex').read()
    report = template.replace(
        '{{parameters}}', latex_itemize_properties(args.simulation_properties)
    ).replace(
        '{{photons}}', itemize_photons(beamlets)
    )
    path = os.path.join(args.output_dir, 'report.tex')
    open(path, 'w').write(report)
    latex_args, rest = latexmake.arg_parser().parse_known_args()
    os.chdir(args.output_dir)
    latexmake.LatexMaker('report', latex_args).run()
    os.chdir('..')
    logger.info('Finished writing report files to {} in {:.2f} seconds'.format(args.output_dir, time.time() - start))
