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
import numpy as np

import report
import egsinp
import grace
import collimator_analyzer
import dose_contours
import py3ddose

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
    parser.add_argument('--target-distance', type=float, default=None,
                        help='Distance from end of collimator to isocenter')
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
    for subfolder in ['dose', 'arc_dose']:
        os.makedirs(os.path.join(args.output_dir, subfolder), exist_ok=True)

    args.egs_home = os.path.abspath(os.path.join(
        os.environ['HEN_HOUSE'], '../egs_home/'))
    logger.info('egs_home is {}'.format(args.egs_home))

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
        logger.info('{} of {} simulations complete, {:.2f} mimnutes remaining'.format(
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
            weight = 1 + (y * y) / (args.target_distance * args.target_distance)
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
    beam_simulations(folder, args.pegs4, simulations)
    index = len(simulations) // 2
    egslst = os.path.join(folder, simulations[index]['egsinp'].replace('.egsinp', '.egslst'))
    shutil.copy(egslst, os.path.join(args.output_dir, 'source{}.egslst'.format(index)))
    _egsinp = os.path.join(folder, simulations[index]['egsinp'])
    shutil.copy(_egsinp, os.path.join(args.output_dir, 'source{}.egsinp'.format(index)))
    if args.rotate:
        beamlets = beam_rotations(beamlets)
    return beamlets, histories


def build_filter(args):
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
    template['isourc'] = '21'
    template['iqin'] = '0'
    for k in ['nrcycl', 'iparallel', 'parnum', 'isrc_dbs', 'rsrc_dbs', 'ssdrc_dbs', 'zsrc_dbs']:
        template[k] = '0'
    template['init_icm'] = 1
    return template


def filter_source(beamlets, _filter, args):
    logger.info('Filtering')
    name = 'FILTR'
    folder = os.path.join(args.egs_home, 'BEAM_{}'.format(name))
    if not os.path.exists(folder):
        beam_build(args.egs_home, name, _filter['cms'])
    filtered_beamlets = []
    simulations = []
    for i, beamlet in enumerate(beamlets):
        _filter['ncase'] = beamlet['stats']['total_particles']
        _filter['spcnam'] = os.path.join('../', 'BEAM_RFLCT', os.path.basename(beamlet['phsp']))
        egsinp_str = egsinp.unparse_egsinp(_filter)
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


def build_collimator(args):
    template = get_egsinp(args.egsinp_template)
    template['cms'] = []
    collimator = get_egsinp(args.collimator)
    template['cms'] = [cm for cm in collimator['cms'] if cm['type'] == 'BLOCK']
    if not template['cms']:
        raise ValueError('No BLOCK CMs found in collimator')
    # for collimators that are part of a larger egsinp simulation file
    zoffset = template['cms'][0]['zmin']
    if not args.target_distance:
        # target distance is measured from the end of the collimator
        args.target_distance = template['cms'][0]['zfocus'] - zoffset - template['cms'][-1]['zmax']
        logger.info('Inferring target distance of {} cm'.format(args.target_distance))
    for block in template['cms']:
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
    return template


def collimate(beamlets, collimator, args):
    logger.info('Collimating')
    name = 'CLMT{}'.format(len(collimator['cms']))
    folder = os.path.join(args.egs_home, 'BEAM_{}'.format(name))
    if not os.path.exists(folder):
        beam_build(args.egs_home, name, collimator['cms'])
    simulations = []
    collimated_beamlets = []
    for i, beamlet in enumerate(beamlets):
        collimator['ncase'] = beamlet['stats']['total_particles']
        collimator['spcnam'] = '../BEAM_FILTR/' + os.path.basename(beamlet['phsp'])
        egsinp_str = egsinp.unparse_egsinp(collimator)
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
    templates = {
        'stationary': open(args.dose_egsinp).read(),
        'arc': open(args.arc_dose_egsinp).read()
    }
    folder = os.path.join(args.egs_home, 'dosxyznrc')
    doses = {}
    simulations = {}
    for stage in ['stationary', 'arc']:
        for i, beamlet in enumerate(beamlets):
            # run two simulations, normal and arced
            context = {
                'egsphant_path': os.path.join(SCRIPT_DIR, args.phantom),
                'phsp_path': beamlet['phsp'],
                'ncase': beamlet['stats']['total_photons'] * (args.dose_recycle + 1),
                'nrcycl': args.dose_recycle,
                'n_split': args.dose_photon_splitting,
                'dsource': args.target_distance,
                'phicol': 90,
                'x': args.target_x,
                'y': args.target_y,
                'z': args.target_z,
                'idat': 1,
                # only for stationary
                'theta': 180,
                'phi': 0
            }
            egsinp_str = templates[stage].format(**context)
            md5 = beamlet['hash'].copy()
            md5.update(egsinp_str.encode('utf-8'))
            base = md5.hexdigest()
            inp = '{}.egsinp'.format(base)
            inp_path = os.path.join(folder, inp)
            open(inp_path, 'w').write(egsinp_str)
            dose_filename = '{}.3ddose'.format(base)
            dose_path = os.path.join(folder, dose_filename)
            simulations.setdefault(stage, []).append({
                'egsinp': inp,
                'dose': dose_path
            })
            doses.setdefault(stage, []).append({
                'dose': dose_path,
                'hash': md5
            })
        dose_simulations(folder, args.pegs4, simulations[stage])
        index = len(simulations) // 2
        egslst = os.path.join(folder, simulations[stage][index]['egsinp'].replace('.egsinp', '.egslst'))
        shutil.copy(egslst, os.path.join(args.output_dir, '{}_dose{}.egslst'.format(stage, index)))
        _egsinp = os.path.join(folder, simulations[stage][index]['egsinp'])
        shutil.copy(_egsinp, os.path.join(args.output_dir, '{}_dose{}.egsinp'.format(stage, index)))
        for i, dose in enumerate(doses[stage]):
            ipath = dose['dose'] + '.npz'
            opath = os.path.join(args.output_dir, '{}_dose/{}_dose{}.3ddose.npz'.format(stage, stage, i))
            shutil.copy(ipath, opath)
    return doses


def slow_dose(beamlets, args):
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
                'dsource': args.target_distance,
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


def combine_fast_doses(doses):
    logger.info('Combining doses')
    result = {}
    weights = {
        'weighted': np.ones(len(doses)),
        'arc_weighted': np.ones(len(doses))
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
        result[slug] = path
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
    source, histories = generate_source(args)
    beamlets['source'] = beamlet_stats(source)
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
    skin_target = {}
    for slug, path in combined_doses.items():
        contours[slug] = dose_contours.plot(args.phantom, path, target, args.output_dir, slug)
        dose = py3ddose.read_3ddose(path)
        conformity[slug] = py3ddose.paddick(dose, target)
        skin_target[slug] = py3ddose.simplified_skin_to_target_ratio(dose, target)

    # we take the plane
    _contours = {}
    for stage, planes in contours.items():
        for contour in planes:
            _contours.setdefault(contour['plane'], []).append(contour)

    photons = {}
    for stage in ['source', 'filter', 'collimator']:
        photons[stage] = sum([beamlet['stats']['total_photons'] for beamlet in beamlets[stage]])

    data = {
        '_filter': _filter,
        'collimator': collimator,
        'collimator_stats': collimator_analyzer.analyze(collimator),
        'beamlets': beamlets,
        'phsp': phsp,
        'plots': grace.make_plots(args.output_dir, phsp, args.plot_config),
        'contours': contours,
        'skin_distance': args.target_distance - abs(args.target_z),
        'ci': conformity,
        'st': skin_target,
        'electrons': histories,
        'photons': photons
    }
    report.generate(data, args)

    logger.info('Finished in {:.2f} seconds, output to {}'.format(
        time.time() - start, args.output_dir))
