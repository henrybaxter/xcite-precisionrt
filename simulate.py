import datetime
import time
import shutil
import platform
import argparse
import sys
from subprocess import run as run_process, PIPE
import os
import math
import hashlib
import json
import functools
import logging

from pathos.pools import ProcessPool as Pool
from pathos.helpers import cpu_count

import latexmake
import egsinp
import grace
import py3ddose

logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def parse_args():
    parser = argparse.ArgumentParser()

    # common
    parser.add_argument('name')
    parser.add_argument('--egsinp-template', default='template.egsinp')
    parser.add_argument('--pegs4', default='allkV')
    parser.add_argument('--rmax', type=float, default=50.0,
                        help='Max extent of all component modules')

    # source arguments
    parser.add_argument('--beam-width', type=float, default=0.2,
                        help='Beam width (y) in cm')
    parser.add_argument('--beam-height', type=float, default=0.5,
                        help='Beam height (z) in cm')
    parser.add_argument('--beam-distance', type=float, default=10.0,
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
    parser.add_argument('--phantom-target-distance', type=float, default=40.0,
                        help='Distance from collimator to phantom target')
    parser.add_argument('--beam-weighting', action='store_true',
                        help='Weight beams by r^2/r\'^2')
    #   given
    parser.add_argument('--collimator',
                        help='Input egsinp path or use stamped values')
    #   preprocess filter phase space
    parser.add_argument('--rotate', action='store_true',
                        help='Rotate phase space files before collimator?')

    # dose
    parser.add_argument('--phantom', default='cylindricalp.egsphant',
                        help='.egsphant file')
    parser.add_argument('--dos-egsinp', default='dose_template.egsinp',
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
    for subfolder in ['dose', 'arc_dose']:
        os.makedirs(os.path.join(args.output_dir, subfolder), exist_ok=True)

    args.egs_home = os.path.abspath(os.path.join(
        os.environ['HEN_HOUSE'], '../egs_home/'))
    logger.info('egs_home is {}'.format(args.egs_home))

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
            'regions_per_block': None,
            'interpolating_blocks': None
        },
    }
    return args


def write_specmodule(egs_home, name, cms):
    logger.info('Writing spec module for {}'.format(name))
    types = []
    identifiers = []
    for cm in cms:
        types.append(cm['type'])
        identifiers.append(cm['identifier'])
    types_line = ' CM names:  {}'.format(' '.join(types))
    identifiers_line = ' Identifiers:  {}'.format(' '.join(identifiers))
    path = os.path.join(egs_home,
                        'beamnrc/spec_modules',
                        '{}.module'.format(name))
    open(path, 'w').write('\n'.join([types_line, identifiers_line]))
    logger.info('Spec module written')


def beam_build(egs_home, name, cms):
    logger.info('Building {}'.format(name))
    start = time.time()
    write_specmodule(egs_home, name, cms)
    logger.info('Running beam_build.exe')
    run_command(['beam_build.exe', name])
    logger.info('Running make')
    run_command(['make'], cwd=os.path.join(egs_home, 'BEAM_{}'.format(name)))
    if platform.system() == 'Darwin':
        run_command([
            'install_name_tool',
            '-change',
            '../egs++/dso/osx/libiaea_phsp.dylib',
            '/Users/henrybaxter/projects/EGSnrc/HEN_HOUSE/egs++/dso/osx/libiaea_phsp.dylib',
            '/Users/henrybaxter/projects/EGSnrc/egs_home/bin/osx/BEAM_{}'.format(name)]
        )
    elapsed = time.time() - start
    logger.info('{} built in {} seconds'.format(name, elapsed))


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
    to_simulate = []
    for simulation in simulations:
        # we only check for the compressed version, since we delete the 3ddose file to save space
        if not os.path.exists(simulation['dose'] + '.npz'):
            to_simulate.append(simulation)
    logger.info('Reusing {} and running {} dose calculations'.format(
        len(simulations) - len(to_simulate), len(to_simulate)))
    dose = functools.partial(dose_simulation, folder, pegs4)
    pool = Pool(cpu_count() - 1)
    start = time.time()
    for i, result in enumerate(pool.imap(dose, to_simulate)):
        elapsed = time.time() - start
        portion_complete = (i + 1) / len(to_simulate)
        estimated_remaining = elapsed / portion_complete
        print('{} of {} simulations complete, {:.2f} mimnutes remaining'.format(
            i + 1, len(to_simulate), estimated_remaining / 60))


def dose_simulation(folder, pegs4, simulation):
    try:
        os.remove(simulation['dose'])
    except IOError:
        pass
    command = ['dosxyznrc', '-p', pegs4, '-i', simulation['egsinp']]
    logger.info('Running "{}"'.format(' '.join(command)))
    result = run_process(command, stdout=PIPE, stderr=PIPE, cwd=folder)
    out = result.stdout.decode('utf-8') + result.stderr.decode('utf-8')
    if result.returncode != 0 or 'ERROR' in out:
        logger.error('Could not run dosxyz on {}'.format(simulation['egsinp']))
        logger.error(result.args)
        logger.error(out)
    # logger.info(result)
    egslst = os.path.join(folder, simulation['egsinp'].replace('.egsinp', '.egslst'))
    logger.info('Writing to {}'.format(egslst))
    open(egslst, 'w').write(out)
    if 'Warning' in out:
        logger.info('Warning in {}'.format(egslst))
    py3ddose.read_3ddose(simulation['dose'])
    os.remove(simulation['dose'])


def map_with_progress(function, items, cpus=None):
    if cpus is None:
        cpus = cpu_count() - 1
    pool = Pool(cpus)
    start = time.time()
    for i, result in enumerate(pool.imap(function, items)):
        elapsed = time.time() - start
        portion_complete = (i + 1) / len(items)
        estimated_remaining = elapsed / portion_complete - elapsed
        logger.info('{} of {} items complete, {}:{} remaining'.format(
            i + 1, len(items), int(estimated_remaining) // 60, int(estimated_remaining) % 60))


def beam_simulations(folder, pegs4, simulations):
    to_simulate = []
    for simulation in simulations:
        if not os.path.exists(simulation['phsp']):
            to_simulate.append(simulation)
    logger.info('Reusing {} and running {} simulations'.format(
        len(simulations) - len(to_simulate), len(to_simulate)))
    simulate = functools.partial(beam_simulation, folder, pegs4)
    map_with_progress(simulate, to_simulate)
    if 'translate_y' in simulations[0]:
        logger.info('Translating phase space files')
        map_with_progress(beam_translate, to_simulate)


def beam_translate(item):
    y = '(' + str(item['translate_y']) + ')'
    command = ['beamdpr', 'translate', '-i', item['phsp'], '-y', y]
    run_command(command)


def beam_simulation(folder, pegs4, item):
    try:
        os.remove(item['phsp'])
        logger.info('Removed old phase space file {}'.format(item['phsp']))
    except IOError:
        pass
    executable = os.path.basename(os.path.normpath(folder))
    command = [executable, '-p', pegs4, '-i', item['egsinp']]
    result = None
    logger.info('Running "{}"'.format(' '.join(command)))
    result = run_process(command, stdout=PIPE, stderr=PIPE, cwd=folder)
    out = result.stdout.decode('utf-8') + result.stderr.decode('utf-8')
    if result.returncode != 0 or 'ERROR' in out:
        logger.error('Could not run simulation on {}'.format(item['egsinp']))
        logger.error(result.args)
        logger.error(out)


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


def run_command(command, **kwargs):
    result = run_process(command, stdout=PIPE, stderr=PIPE, **kwargs)
    if result.returncode != 0:
        logger.error('Command failed: "{}"'.format(' '.join(command)))
        logger.error(result.stdout.decode('utf-8'))
        logger.error(result.stderr.decode('utf-8'))
        sys.exit(1)
    return result.stdout.decode('utf-8')


def beamlet_stats(beamlets):
    logger.info('Gathering stats for {} beamlets'.format(len(beamlets)))
    for beamlet in beamlets:
        command = ['beamdpr', 'stats', '--format=json', beamlet['phsp']]
        beamlet['stats'] = json.loads(run_command(command))
    return beamlets


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


def generate_source(args):
    logger.info('Generating source with {} histories'.format(args.histories))
    y_values = generate_y(args.target_length, args.beam_width + args.beam_gap)
    template = get_egsinp(args.egsinp_template)

    # rebuild mortran if requested
    name = 'RFLCT'
    folder = os.path.join(args.egs_home, 'BEAM_{}'.format(name))
    if not os.path.exists(folder):
        beam_build(args.egs_home, name, template['cms'])
    template['ncase'] = args.histories // len(y_values)
    template['ybeam'] = args.beam_width / 2
    template['zbeam'] = args.beam_height / 2
    logger.info('Using {} histories per beamlet'.format(template['ncase']))
    xtube = template['cms'][0]
    xtube['rmax_cm'] = args.rmax
    xtube['anglei'] = args.target_angle

    logger.info('Creating egsinp files')
    beamlets = []
    simulations = []
    histories = 0
    for i, y in enumerate(y_values):
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
        inp_path = os.path.join(folder, inp)
        open(inp_path, 'w').write(egsinp_str)
        phsp = os.path.join(folder, '{}.egsphsp1'.format(base))
        simulations.append({
            'egsinp': inp,
            'phsp': phsp,
            'translate_y': y
        })
        beamlets.append({
            'phsp': phsp,
            'hash': md5
        })
    args.simulation_properties['source']['histories'] = histories
    beam_simulations(folder, args.pegs4, simulations)
    index = len(simulations) // 2
    egslst = os.path.join(folder, simulations[index]['egsinp'].replace('.egsinp', '.egslst'))
    shutil.copy(egslst, os.path.join(args.output_dir, 'source{}.egslst'.format(index)))
    _egsinp = os.path.join(folder, simulations[index]['egsinp'])
    shutil.copy(_egsinp, os.path.join(args.output_dir, 'source{}.egsinp'.format(index)))
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
    name = 'FILTR'
    folder = os.path.join(args.egs_home, 'BEAM_{}'.format(name))
    if not os.path.exists(folder):
        beam_build(args.egs_home, name, template['cms'])
    filtered_beamlets = []
    simulations = []
    for i, beamlet in enumerate(beamlets):
        template['ncase'] = beamlet['stats']['total_particles']
        template['spcnam'] = os.path.join('../', 'BEAM_RFLCT', os.path.basename(beamlet['phsp']))
        egsinp_str = egsinp.unparse_egsinp(template)
        md5 = beamlet['hash'].copy()
        md5.update(egsinp_str.encode('utf-8'))
        base = md5.hexdigest()
        inp = '{}.egsinp'.format(base)
        inp_path = os.path.join(folder, inp)
        open(inp_path, 'w').write(egsinp_str)
        phsp = os.path.join(folder, '{}.egsphsp1'.format(base))
        simulations.append({
            'egsinp': inp,  # filename
            'phsp': phsp,  # full path
        })
        filtered_beamlets.append({
            'phsp': phsp,
            'hash': md5
        })
    beam_simulations(folder, args.pegs4, simulations)
    index = len(simulations) // 2
    egslst = os.path.join(folder, simulations[index]['egsinp'].replace('.egsinp', '.egslst'))
    shutil.copy(egslst, os.path.join(args.output_dir, 'filter{}.egslst'.format(index)))
    _egsinp = os.path.join(folder, simulations[index]['egsinp'])
    shutil.copy(_egsinp, os.path.join(args.output_dir, 'filter{}.egsinp'.format(index)))
    return filtered_beamlets


def beam_rotate(angle, item):
    run_command(['beamdpr', 'rotate', item['input'], item['output'], '-a', angle])


def beam_rotations(beamlets):
    logger.info("Rotating beamlets")
    to_rotate = []
    for beamlet in beamlets:
        beamlet['hash'].update('zrotatep=90'.encode('utf-8'))
        filename = beamlet['hash'].hexdigest() + '.egsphsp1'
        phsp = os.path.join(os.path.dirname(beamlet['phsp']), filename)
        if not os.path.exists(phsp):
            to_rotate.append({
                'input': beamlet['phsp'],
                'output': phsp
            })
        beamlet['phsp'] = phsp
    logger.info('Reusing {} and running {} rotations'.format(
        len(beamlets) - len(to_rotate), len(to_rotate)))
    rotate = functools.partial(beam_rotate, str(math.pi / 2))
    map_with_progress(rotate, to_rotate)
    return beamlets


def collimate(beamlets, args):
    logger.info('Collimating')
    template = get_egsinp(args.egsinp_template)
    template['cms'] = []
    collimator = get_egsinp(args.collimator)
    template['cms'] = [cm for cm in collimator['cms'] if cm['type'] == 'BLOCK']
    if not template['cms']:
        raise ValueError('No BLOCK CMs found in collimator')
    zoffset = template['cms'][0]['zmin']
    for block in template['cms']:
        if zoffset is None:
            zoffset = block['zmin']
        block['zmin'] -= zoffset
        block['zmax'] -= zoffset
        block['zfocus'] -= zoffset
        block['rmax_cm'] = args.rmax
    template['isourc'] = '21'
    template['iqin'] = '0'
    template['default_medium'] = 'Air_516kV'
    template['nsc_planes'] = '1'
    template['init_icm'] = 1
    template['nrcycl'] = 0
    template['iparallel'] = 0
    template['parnum'] = 0
    template['isrc_dbs'] = 0
    template['rsrc_dbs'] = 0
    template['ssdrc_dbs'] = 0
    template['zsrc_dbs'] = 0
    template['scoring_planes'] = [{
        'cm': len(template['cms']),  # the LAST block of the collimator
        'mzone_type': 1,
        'nsc_zones': 1,
        'zones': [args.rmax]
    }]
    name = 'CLMT{}'.format(len(template['cms']))
    folder = os.path.join(args.egs_home, 'BEAM_{}'.format(name))
    if not os.path.exists(folder):
        beam_build(args.egs_home, name, template['cms'])
    args.simulation_properties['collimator']['interpolating_blocks'] = len(template['cms'])
    first_block = template['cms'][0]
    last_block = template['cms'][-1]
    args.simulation_properties['collimator']['length'] = last_block['zmax'] - first_block['zmin']
    args.simulation_properties['collimator']['regions_per_block'] = len(first_block['regions'])
    simulations = []
    collimated_beamlets = []
    for i, beamlet in enumerate(beamlets):
        template['ncase'] = beamlet['stats']['total_particles']
        template['spcnam'] = '../BEAM_FILTR/' + os.path.basename(beamlet['phsp'])
        egsinp_str = egsinp.unparse_egsinp(template)
        md5 = beamlet['hash'].copy()
        md5.update(egsinp_str.encode('utf-8'))
        base = md5.hexdigest()
        inp = '{}.egsinp'.format(base)
        inp_path = os.path.join(folder, inp)
        open(inp_path, 'w').write(egsinp_str)
        phsp = os.path.join(folder, '{}.egsphsp1'.format(base))
        simulations.append({
            'egsinp': inp,  # filename
            'phsp': phsp,  # full path
        })
        collimated_beamlets.append({
            'phsp': phsp,
            'hash': md5
        })
    beam_simulations(folder, args.pegs4, simulations)
    index = len(simulations) // 2
    egslst = os.path.join(folder, simulations[index]['egsinp'].replace('.egsinp', '.egslst'))
    shutil.copy(egslst, os.path.join(args.output_dir, 'collimator{}.egslst'.format(index)))
    _egsinp = os.path.join(folder, simulations[index]['egsinp'])
    shutil.copy(_egsinp, os.path.join(args.output_dir, 'collimator{}.egsinp'.format(index)))
    return collimated_beamlets


def dose_angles(args):
    # recall that 180 theta is center
    # and we do (theta, phi)
    angles = [(180, 0)]
    angular_increment = 5  # degrees
    angular_sweep = 120  # degrees
    n_angles_per_side = int(angular_sweep / angular_increment) // 2
    for j in range(n_angles_per_side):
        angles.append((180 - (j + 1) * angular_increment, 0))
    for j in range(n_angles_per_side):
        angles.append((180 - (j + 1) * angular_increment, 180))
    return angles


def fast_dose(beamlets, args):
    logger.info('Fast dosing')
    template = open(args.dos_egsinp).read()
    arc_template = open(args.arc_dose_egsinp).read()
    folder = os.path.join(args.egs_home, 'dosxyznrc')
    doses = []
    arc_doses = []
    simulations = []
    arc_simulations = []
    for i, beamlet in enumerate(beamlets):
        # run two simulations, normal and arced
        context = {
            'egsphant_path': os.path.join(SCRIPT_DIR, args.phantom),
            'phsp_path': beamlet['phsp'],
            'ncase': beamlet['stats']['total_photons'] * (args.dose_recycle + 1),
            'nrcycl': args.dose_recycle,
            'n_split': args.dose_photon_splitting,
            'dsource': args.phantom_target_distance,
            'phicol': 90,
            'x': 0,
            'y': 20,
            'z': -10,
            'idat': 1  # do NOT output intermediate files (.egsdat?)
        }
        dose_context = context.copy()
        dose_context['theta'] = 180
        dose_context['phi'] = 0
        arc_context = context.copy()
        egsinp_str = template.format(**dose_context)
        arc_egsinp_str = arc_template.format(**arc_context)
        md5 = beamlet['hash'].copy()
        arc_md5 = beamlet['hash'].copy()
        base = md5.hexdigest()
        arc_base = arc_md5.hexdigest()
        inp = '{}.egsinp'.format(base)
        arc_inp = '{}.egsinp'.format(arc_base)
        inp_path = os.path.join(folder, inp)
        arc_inp_path = os.path.join(folder, arc_inp)
        open(inp_path, 'w').write(egsinp_str)
        open(arc_inp_path, 'w').write(arc_egsinp_str)
        dose_filename = '{}.3ddose'.format(base)
        arc_dose_filename = '{}.3ddose'.format(arc_base)
        dose_path = os.path.join(folder, dose_filename)
        arc_dose_path = os.path.join(folder, arc_dose_filename)
        simulations.append({
            'egsinp': inp,
            'dose': dose_path
        })
        arc_simulations.append({
            'egsinp': arc_inp,
            'dose': arc_dose_path
        })
        doses.append({
            'dose': dose_path,
            'hash': md5
        })
        arc_doses.append({
            'dose': arc_dose_path,
            'hash': md5
        })
    dose_simulations(folder, args.pegs4, simulations)
    dose_simulations(folder, args.pegs4, arc_simulations)
    index = len(simulations) // 2
    egslst = os.path.join(folder, simulations[index]['egsinp'].replace('.egsinp', '.egslst'))
    shutil.copy(egslst, os.path.join(args.output_dir, 'dose{}.egslst'.format(index)))
    _egsinp = os.path.join(folder, simulations[index]['egsinp'])
    shutil.copy(_egsinp, os.path.join(args.output_dir, 'dose{}.egsinp'.format(index)))
    egslst = os.path.join(folder, arc_simulations[index]['egsinp'].replace('.egsinp', '.egslst'))
    shutil.copy(egslst, os.path.join(args.output_dir, 'arc_dose{}.egslst'.format(index)))
    _egsinp = os.path.join(folder, arc_simulations[index]['egsinp'])
    shutil.copy(_egsinp, os.path.join(args.output_dir, 'arc_dose{}.egsinp'.format(index)))
    for i, dose in enumerate(doses):
        ipath = dose['dose'] + '.npz'
        opath = os.path.join(args.output_dir, 'dose/dose{}.3ddose.npz'.format(i))
        shutil.copy(ipath, opath)
    for i, dose in enumerate(arc_doses):
        ipath = dose['dose'] + '.npz'
        opath = os.path.join(args.output_dir, 'arc_dose/arc_dose{}.3ddose.npz'.format(i))
        shutil.copy(ipath, opath)
    return doses, arc_doses


def dose(beamlets, args):
    logger.info('Slow dosing')
    template = open(args.dos_egsinp).read()
    folder = os.path.join(args.egs_home, 'dosxyznrc')
    dose_contributions = []
    simulations = []
    for i, beamlet in enumerate(beamlets):
        angled_dose_contributions = {}
        for j, (theta, phi) in enumerate(dose_angles(args)):
            kwargs = {
                'egsphant_path': os.path.join(SCRIPT_DIR, args.phantom),
                'phsp_path': beamlet['phsp'],
                'ncase': beamlet['stats']['total_photons'] * (args.dose_recycle + 1),
                'nrcycl': args.dose_recycle,
                'n_split': args.dose_photon_splitting,
                'dsource': args.phantom_target_distance,
                'theta': theta,
                'phi': phi,
                'phicol': 90
            }
            logger.info('Dose using each particle {} times so {} histories'.format(
                kwargs['nrcycl'] + 1, kwargs['ncase']))
            egsinp_str = template.format(**kwargs)
            md5 = beamlet['hash'].copy()
            md5.update(egsinp_str.encode('utf-8'))
            base = md5.hexdigest()
            inp = '{}.egsinp'.format(base)
            inp_path = os.path.join(folder, inp)
            open(inp_path, 'w').write(egsinp_str)
            dose_filename = '{}.3ddose'.format(base)
            dose_path = os.path.join(folder, dose_filename)
            simulations.append({
                'egsinp': inp,
                'dose': dose_path
            })
            angled_dose_contributions[(theta, phi)] = {
                'dose': dose_path,
                'hash': md5
            }
        dose_contributions.append(angled_dose_contributions)
    shutil.copy(inp_path, os.path.join(args.output_dir, 'last_dose.egsinp'))
    dose_simulations(folder, args.pegs4, simulations)
    egslst = inp_path.replace('.egsinp', '.egslst')
    shutil.copy(egslst, os.path.join(args.output_dir, 'last_dose.egslst'))
    for i, beamlet_contribution in enumerate(dose_contributions):
        for (theta, phi), contribution in beamlet_contribution.items():
            slug = '{}_{}_{}'.format(i, theta, phi)
            ipath = contribution['dose'] + '.npz'
            opath = os.path.join(args.output_dir, 'dose/dose{}.3ddose.npz'.format(slug))
            shutil.copy(ipath, opath)
    return dose_contributions


def combine_doses(contributions):
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
    lines.append('\t\item Efficiency: {:.2f} electrons generates one photon'.format(overall))
    return '\n'.join(lines)


def configure_logging(args):
    revision = run_command(['git', 'rev-parse', '--short', 'HEAD']).strip()
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


def write_report(args, beamlets):
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
    logger.info('Report written to {} in {:.2f} seconds'.format(
        args.output_dir, time.time() - start))


def log_args(args):
    printable_args = [(k, v) for k, v in sorted(args.__dict__.items())]
    pretty_args = '\n'.join(['\t{}: {}'.format(k, v) for k, v in printable_args])
    logger.info('Arguments: \n{}'.format(pretty_args))


if __name__ == '__main__':
    start = time.time()
    args = parse_args()
    configure_logging(args)
    log_args(args)

    phsp = {}
    beamlets = {}
    beamlets['source'] = beamlet_stats(generate_source(args))
    if args.rotate:
        beamlets['source'] = beam_rotations(beamlets['source'])
    phsp['source'] = sample_combine(beamlets['source'])
    shutil.copy(phsp['source'], os.path.join(args.output_dir, 'sampled_source.egsphsp1'))

    beamlets['filter'] = beamlet_stats(filter_source(beamlets['source'], args))
    phsp['filter'] = sample_combine(beamlets['filter'])
    shutil.copy(phsp['filter'], os.path.join(args.output_dir, 'sampled_filter.egsphsp1'))

    beamlets['collimator'] = beamlet_stats(collimate(beamlets['filter'], args))
    phsp['collimator'] = sample_combine(beamlets['collimator'], desired=100000000)
    shutil.copy(phsp['collimator'], os.path.join(args.output_dir, 'sampled_collimator.egsphsp1'))

    # dose_contributions = dose(beamlets['collimator'], args)
    # combine_doses(dose_contributions)
    doses, arc_doses = fast_dose(beamlets['collimator'], args)
    paths = [dose['path'] + '.npz' for dose in doses]
    opath = os.path.join(args.output_dir, 'dose.3ddose')
    py3ddose.combine_3ddose(paths, opath)
    py3ddose.read_3ddose(opath)
    paths = [dose['path'] + '.npz' for dose in arc_doses]
    opath = os.path.join(args.output_dir, 'arc_dose.3ddose')
    py3ddose.combine_3ddose(paths, opath)
    py3ddose.read_3ddose(opath)

    plots = grace_plot(args.output_dir, phsp, args)

    write_report(args, beamlets)
